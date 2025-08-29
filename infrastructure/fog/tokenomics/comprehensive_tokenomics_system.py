#!/usr/bin/env python3
"""
Comprehensive Tokenomics Deployment System

This module provides complete tokenomics deployment and economic systems for the FOG token ecosystem:
- FOG token economics for fog computing resources
- Credit systems for resource usage tracking
- Reward mechanisms for compute contributions
- Staking and governance token integration
- Economic incentive mechanisms
- Token supply management and inflation control
- Liquidity management and market making
- Cross-chain bridge integration

Key Features:
- Multi-tier token economy (FOG, veFOG, credit tokens)
- Dynamic reward rates based on network utilization
- Staking mechanisms with time-locked rewards
- Governance token voting power calculation
- Economic incentive optimization
- Anti-whale mechanisms and fair distribution
- Integration with fog computing infrastructure
- Compliance with tokenomics best practices
"""

import asyncio
import json
import logging
import math
import secrets
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Set high precision for token calculations
getcontext().prec = 28

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenType(Enum):
    """Types of tokens in the ecosystem."""
    FOG = "FOG"  # Main utility token
    VE_FOG = "veFOG"  # Vote-escrowed FOG for governance
    CREDIT = "CREDIT"  # Resource usage credits
    COMPUTE = "COMPUTE"  # Compute contribution rewards
    LIQUIDITY = "LIQUIDITY"  # LP tokens for AMM


class TransactionType(Enum):
    """Types of token transactions."""
    TRANSFER = "transfer"
    REWARD = "reward"
    STAKE = "stake"
    UNSTAKE = "unstake"
    SLASH = "slash"
    BURN = "burn"
    MINT = "mint"
    GOVERNANCE_REWARD = "governance_reward"
    COMPUTE_REWARD = "compute_reward"
    LIQUIDITY_REWARD = "liquidity_reward"


class StakingTier(Enum):
    """Staking tier levels with different benefits."""
    BRONZE = "bronze"    # 30-day lock, 5% APY
    SILVER = "silver"    # 90-day lock, 8% APY
    GOLD = "gold"        # 180-day lock, 12% APY
    PLATINUM = "platinum"  # 365-day lock, 18% APY
    DIAMOND = "diamond"  # 730-day lock, 25% APY


class TokenAccount(BaseModel):
    """Token account model with multi-token balances."""
    account_id: str
    address: str
    
    # Token balances
    fog_balance: Decimal = Decimal("0")
    ve_fog_balance: Decimal = Decimal("0")
    credit_balance: Decimal = Decimal("0")
    compute_balance: Decimal = Decimal("0")
    
    # Staking information
    total_staked: Decimal = Decimal("0")
    staking_positions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Governance metrics
    voting_power: Decimal = Decimal("0")
    governance_participation: int = 0
    
    # Economic metrics
    total_earned: Decimal = Decimal("0")
    total_spent: Decimal = Decimal("0")
    compute_contributions: Decimal = Decimal("0")
    
    # Account status
    created_at: datetime
    last_activity: datetime
    active: bool = True
    kyc_verified: bool = False
    tier_level: str = "basic"  # basic, verified, institutional
    
    def get_effective_voting_power(self) -> Decimal:
        """Calculate effective voting power based on staking and participation."""
        base_power = self.ve_fog_balance
        
        # Participation bonus (up to 20%)
        participation_bonus = min(Decimal("0.2"), self.governance_participation * Decimal("0.01"))
        
        # Long-term staking bonus
        long_term_bonus = Decimal("0")
        for position in self.staking_positions:
            if position["lock_days"] >= 365:
                long_term_bonus += position["amount"] * Decimal("0.1")
        
        return base_power * (Decimal("1") + participation_bonus) + long_term_bonus


class TokenTransaction(BaseModel):
    """Token transaction record."""
    tx_id: str
    tx_type: TransactionType
    token_type: TokenType
    from_account: str
    to_account: str
    amount: Decimal
    fee: Decimal = Decimal("0")
    
    # Transaction metadata
    timestamp: datetime
    block_height: int = 0
    gas_used: int = 0
    
    # Additional data
    memo: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Status tracking
    confirmed: bool = False
    confirmation_count: int = 0


class StakingPosition(BaseModel):
    """Individual staking position."""
    position_id: str
    account_id: str
    tier: StakingTier
    amount: Decimal
    lock_days: int
    apy_rate: Decimal
    
    # Timing
    staked_at: datetime
    unlock_at: datetime
    last_reward_claim: datetime
    
    # Rewards
    rewards_earned: Decimal = Decimal("0")
    pending_rewards: Decimal = Decimal("0")
    
    # Status
    active: bool = True
    early_withdrawal: bool = False
    penalty_applied: Decimal = Decimal("0")


class EconomicMetrics(BaseModel):
    """System-wide economic metrics."""
    timestamp: datetime
    
    # Supply metrics
    total_supply: Decimal
    circulating_supply: Decimal
    staked_supply: Decimal
    burned_supply: Decimal
    
    # Network metrics
    active_accounts: int
    daily_transactions: int
    network_utilization: Decimal  # 0-1
    
    # Reward metrics
    daily_rewards_distributed: Decimal
    compute_rewards: Decimal
    staking_rewards: Decimal
    governance_rewards: Decimal
    
    # Price and liquidity (if available)
    token_price_usd: Optional[Decimal] = None
    market_cap_usd: Optional[Decimal] = None
    liquidity_pool_size: Decimal = Decimal("0")
    
    # Economic health indicators
    reward_sustainability_ratio: Decimal  # rewards/inflation
    staking_participation_rate: Decimal
    governance_participation_rate: Decimal


class ComprehensiveTokenomicsSystem:
    """Complete tokenomics deployment and management system."""
    
    def __init__(self, data_dir: str = "./tokenomics_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core data structures
        self.accounts: Dict[str, TokenAccount] = {}
        self.transactions: List[TokenTransaction] = []
        self.staking_positions: Dict[str, StakingPosition] = {}
        
        # Economic parameters
        self.token_config = {
            # Supply parameters
            "max_supply": Decimal("1000000000"),  # 1B FOG tokens
            "initial_supply": Decimal("100000000"),  # 100M initial
            "inflation_rate": Decimal("0.05"),  # 5% annual inflation
            "burn_rate": Decimal("0.01"),  # 1% of fees burned
            
            # Reward parameters
            "base_compute_reward": Decimal("10"),  # FOG per hour
            "network_utilization_multiplier": Decimal("2"),  # Max 2x for high utilization
            "staking_reward_pool": Decimal("5000000"),  # Annual staking rewards
            "governance_reward_pool": Decimal("1000000"),  # Annual governance rewards
            
            # Staking parameters
            "min_stake_amount": Decimal("100"),
            "max_stake_per_account": Decimal("10000000"),
            "early_withdrawal_penalty": Decimal("0.25"),  # 25% penalty
            
            # Fee parameters
            "base_transaction_fee": Decimal("0.01"),
            "compute_usage_fee_rate": Decimal("0.001"),  # Per compute unit
            "governance_fee": Decimal("1"),  # For proposal submission
            
            # Economic controls
            "anti_whale_threshold": Decimal("1000000"),  # 1M FOG voting cap
            "min_voting_balance": Decimal("100"),
            "delegation_fee": Decimal("0.1")
        }
        
        # Staking tier configuration
        self.staking_tiers = {
            StakingTier.BRONZE: {"lock_days": 30, "apy": Decimal("0.05")},
            StakingTier.SILVER: {"lock_days": 90, "apy": Decimal("0.08")},
            StakingTier.GOLD: {"lock_days": 180, "apy": Decimal("0.12")},
            StakingTier.PLATINUM: {"lock_days": 365, "apy": Decimal("0.18")},
            StakingTier.DIAMOND: {"lock_days": 730, "apy": Decimal("0.25")}
        }
        
        # Current network state
        self.current_supply = self.token_config["initial_supply"]
        self.total_staked = Decimal("0")
        self.total_burned = Decimal("0")
        self.network_utilization = Decimal("0.5")  # 50% default utilization
        
        # Economic metrics history
        self.metrics_history: List[EconomicMetrics] = []
        
        # Initialize system
        self._initialize_tokenomics()
    
    def _initialize_tokenomics(self):
        """Initialize the tokenomics system with foundational accounts."""
        logger.info("Initializing comprehensive tokenomics system...")
        
        # Create system accounts
        system_accounts = [
            ("treasury", "0x0000000000000000000000000000000000000001", Decimal("20000000")),  # 20M treasury
            ("rewards_pool", "0x0000000000000000000000000000000000000002", Decimal("30000000")),  # 30M rewards
            ("liquidity_pool", "0x0000000000000000000000000000000000000003", Decimal("10000000")),  # 10M liquidity
            ("development", "0x0000000000000000000000000000000000000004", Decimal("15000000")),  # 15M development
            ("ecosystem", "0x0000000000000000000000000000000000000005", Decimal("25000000"))   # 25M ecosystem
        ]
        
        for account_id, address, initial_balance in system_accounts:
            self.create_account(account_id, address, initial_balance)
        
        # Record initial metrics
        self._record_economic_metrics()
        
        logger.info(f"Initialized tokenomics with {len(self.accounts)} system accounts")
        logger.info(f"Total initial supply: {self.current_supply:,} FOG tokens")
    
    def create_account(self, account_id: str, address: str, 
                      initial_balance: Decimal = Decimal("0")) -> TokenAccount:
        """Create a new token account."""
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")
        
        account = TokenAccount(
            account_id=account_id,
            address=address,
            fog_balance=initial_balance,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.accounts[account_id] = account
        
        # Record transaction if initial balance > 0
        if initial_balance > 0:
            self._record_transaction(
                tx_type=TransactionType.MINT,
                token_type=TokenType.FOG,
                from_account="system",
                to_account=account_id,
                amount=initial_balance,
                metadata={"reason": "account_creation"}
            )
        
        logger.info(f"Created token account: {account_id} with {initial_balance} FOG")
        return account
    
    def transfer_tokens(self, from_account: str, to_account: str, 
                       amount: Decimal, token_type: TokenType = TokenType.FOG,
                       memo: str = None) -> str:
        """Transfer tokens between accounts."""
        if from_account not in self.accounts or to_account not in self.accounts:
            raise ValueError("Both accounts must exist")
        
        sender = self.accounts[from_account]
        receiver = self.accounts[to_account]
        
        # Check balance based on token type
        if token_type == TokenType.FOG and sender.fog_balance < amount:
            raise ValueError("Insufficient FOG balance")
        elif token_type == TokenType.CREDIT and sender.credit_balance < amount:
            raise ValueError("Insufficient CREDIT balance")
        elif token_type == TokenType.COMPUTE and sender.compute_balance < amount:
            raise ValueError("Insufficient COMPUTE balance")
        
        # Calculate fee
        fee = self._calculate_transfer_fee(amount, token_type)
        total_amount = amount + fee
        
        # Check total amount including fee
        if token_type == TokenType.FOG and sender.fog_balance < total_amount:
            raise ValueError("Insufficient balance including fee")
        
        # Execute transfer
        if token_type == TokenType.FOG:
            sender.fog_balance -= total_amount
            receiver.fog_balance += amount
        elif token_type == TokenType.CREDIT:
            sender.credit_balance -= amount
            receiver.credit_balance += amount
        elif token_type == TokenType.COMPUTE:
            sender.compute_balance -= amount
            receiver.compute_balance += amount
        
        # Update activity timestamps
        sender.last_activity = datetime.now()
        receiver.last_activity = datetime.now()
        
        # Record transaction
        tx_id = self._record_transaction(
            tx_type=TransactionType.TRANSFER,
            token_type=token_type,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            fee=fee,
            memo=memo
        )
        
        # Burn fee (deflationary mechanism)
        if fee > 0 and token_type == TokenType.FOG:
            self._burn_tokens(fee, "transfer_fee")
        
        logger.info(f"Transferred {amount} {token_type.value} from {from_account} to {to_account} (fee: {fee})")
        return tx_id
    
    def stake_tokens(self, account_id: str, amount: Decimal, tier: StakingTier) -> str:
        """Stake FOG tokens for rewards and governance power."""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Validate staking amount
        if amount < self.token_config["min_stake_amount"]:
            raise ValueError(f"Minimum stake amount is {self.token_config['min_stake_amount']} FOG")
        
        if account.total_staked + amount > self.token_config["max_stake_per_account"]:
            raise ValueError(f"Maximum stake per account is {self.token_config['max_stake_per_account']} FOG")
        
        if account.fog_balance < amount:
            raise ValueError("Insufficient FOG balance for staking")
        
        # Get tier configuration
        tier_config = self.staking_tiers[tier]
        
        # Create staking position
        position_id = f"stake_{uuid.uuid4().hex[:12]}"
        unlock_date = datetime.now() + timedelta(days=tier_config["lock_days"])
        
        position = StakingPosition(
            position_id=position_id,
            account_id=account_id,
            tier=tier,
            amount=amount,
            lock_days=tier_config["lock_days"],
            apy_rate=tier_config["apy"],
            staked_at=datetime.now(),
            unlock_at=unlock_date,
            last_reward_claim=datetime.now()
        )
        
        self.staking_positions[position_id] = position
        
        # Update account
        account.fog_balance -= amount
        account.total_staked += amount
        account.staking_positions.append({
            "position_id": position_id,
            "tier": tier.value,
            "amount": float(amount),
            "unlock_at": unlock_date.isoformat()
        })
        account.last_activity = datetime.now()
        
        # Convert to veFOG based on lock duration
        ve_fog_amount = self._calculate_ve_fog(amount, tier_config["lock_days"])
        account.ve_fog_balance += ve_fog_amount
        account.voting_power = account.get_effective_voting_power()
        
        # Update global state
        self.total_staked += amount
        
        # Record transaction
        tx_id = self._record_transaction(
            tx_type=TransactionType.STAKE,
            token_type=TokenType.FOG,
            from_account=account_id,
            to_account="staking_pool",
            amount=amount,
            metadata={
                "position_id": position_id,
                "tier": tier.value,
                "lock_days": tier_config["lock_days"],
                "apy_rate": float(tier_config["apy"]),
                "ve_fog_generated": float(ve_fog_amount)
            }
        )
        
        logger.info(f"Staked {amount} FOG for {account_id} in {tier.value} tier (position: {position_id})")
        return position_id
    
    def unstake_tokens(self, account_id: str, position_id: str, force: bool = False) -> Decimal:
        """Unstake tokens from a staking position."""
        if position_id not in self.staking_positions:
            raise ValueError(f"Staking position {position_id} not found")
        
        position = self.staking_positions[position_id]
        account = self.accounts[account_id]
        
        if position.account_id != account_id:
            raise ValueError("Position does not belong to this account")
        
        if not position.active:
            raise ValueError("Staking position is not active")
        
        now = datetime.now()
        is_early_withdrawal = now < position.unlock_at
        
        if is_early_withdrawal and not force:
            raise ValueError(f"Position is locked until {position.unlock_at}")
        
        # Calculate rewards
        rewards = self._calculate_staking_rewards(position)
        
        # Calculate penalty if early withdrawal
        penalty = Decimal("0")
        if is_early_withdrawal:
            penalty = position.amount * self.token_config["early_withdrawal_penalty"]
            position.early_withdrawal = True
            position.penalty_applied = penalty
        
        # Return amount after penalty
        return_amount = position.amount - penalty
        
        # Update account balances
        account.fog_balance += return_amount
        account.fog_balance += rewards  # Add accumulated rewards
        account.total_staked -= position.amount
        account.total_earned += rewards
        
        # Reduce veFOG
        ve_fog_reduction = self._calculate_ve_fog(position.amount, position.lock_days)
        account.ve_fog_balance = max(Decimal("0"), account.ve_fog_balance - ve_fog_reduction)
        account.voting_power = account.get_effective_voting_power()
        
        # Update position
        position.active = False
        position.rewards_earned = rewards
        
        # Update global state
        self.total_staked -= position.amount
        
        # Burn penalty if any
        if penalty > 0:
            self._burn_tokens(penalty, "early_withdrawal_penalty")
        
        # Record transaction
        tx_id = self._record_transaction(
            tx_type=TransactionType.UNSTAKE,
            token_type=TokenType.FOG,
            from_account="staking_pool",
            to_account=account_id,
            amount=return_amount,
            metadata={
                "position_id": position_id,
                "rewards": float(rewards),
                "penalty": float(penalty),
                "early_withdrawal": is_early_withdrawal
            }
        )
        
        logger.info(f"Unstaked {return_amount} FOG + {rewards} rewards for {account_id} "
                   f"(penalty: {penalty}, early: {is_early_withdrawal})")
        
        return return_amount + rewards
    
    def distribute_compute_rewards(self, account_id: str, compute_hours: Decimal, 
                                 quality_multiplier: Decimal = Decimal("1.0")) -> Decimal:
        """Distribute rewards for fog computing contributions."""
        if account_id not in self.accounts:
            self.create_account(account_id, f"0x{secrets.token_hex(20)}")
        
        account = self.accounts[account_id]
        
        # Calculate base reward
        base_reward = compute_hours * self.token_config["base_compute_reward"]
        
        # Apply network utilization multiplier
        utilization_multiplier = Decimal("1") + (self.network_utilization * 
                                               self.token_config["network_utilization_multiplier"])
        
        # Calculate final reward
        total_reward = base_reward * utilization_multiplier * quality_multiplier
        
        # Mint new tokens for rewards (inflationary)
        self._mint_tokens(total_reward, "compute_rewards")
        
        # Distribute to account
        account.fog_balance += total_reward
        account.compute_contributions += compute_hours
        account.total_earned += total_reward
        account.last_activity = datetime.now()
        
        # Record transaction
        tx_id = self._record_transaction(
            tx_type=TransactionType.COMPUTE_REWARD,
            token_type=TokenType.FOG,
            from_account="rewards_pool",
            to_account=account_id,
            amount=total_reward,
            metadata={
                "compute_hours": float(compute_hours),
                "quality_multiplier": float(quality_multiplier),
                "network_utilization": float(self.network_utilization),
                "utilization_multiplier": float(utilization_multiplier)
            }
        )
        
        logger.info(f"Distributed {total_reward} FOG compute rewards to {account_id} "
                   f"for {compute_hours} hours")
        
        return total_reward
    
    def distribute_governance_rewards(self, account_id: str, proposals_participated: int,
                                    votes_cast: int) -> Decimal:
        """Distribute rewards for governance participation."""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Calculate governance rewards based on participation
        proposal_reward = Decimal(proposals_participated) * Decimal("50")  # 50 FOG per proposal
        voting_reward = Decimal(votes_cast) * Decimal("10")  # 10 FOG per vote
        
        # Bonus for consistent participation
        participation_bonus = Decimal("0")
        if account.governance_participation > 10:
            participation_bonus = min(Decimal("100"), account.governance_participation * Decimal("2"))
        
        total_reward = proposal_reward + voting_reward + participation_bonus
        
        # Cap rewards to prevent gaming
        max_monthly_governance_reward = Decimal("1000")
        total_reward = min(total_reward, max_monthly_governance_reward)
        
        if total_reward > 0:
            # Mint tokens for governance rewards
            self._mint_tokens(total_reward, "governance_rewards")
            
            # Distribute to account
            account.fog_balance += total_reward
            account.total_earned += total_reward
            account.governance_participation += proposals_participated + votes_cast
            account.last_activity = datetime.now()
            
            # Update voting power
            account.voting_power = account.get_effective_voting_power()
            
            # Record transaction
            tx_id = self._record_transaction(
                tx_type=TransactionType.GOVERNANCE_REWARD,
                token_type=TokenType.FOG,
                from_account="rewards_pool",
                to_account=account_id,
                amount=total_reward,
                metadata={
                    "proposals_participated": proposals_participated,
                    "votes_cast": votes_cast,
                    "participation_bonus": float(participation_bonus)
                }
            )
            
            logger.info(f"Distributed {total_reward} FOG governance rewards to {account_id}")
        
        return total_reward
    
    def process_periodic_rewards(self) -> Dict[str, Any]:
        """Process periodic staking rewards and economic updates."""
        now = datetime.now()
        total_rewards_distributed = Decimal("0")
        positions_updated = 0
        
        # Process staking rewards
        for position_id, position in self.staking_positions.items():
            if not position.active:
                continue
            
            # Calculate time since last reward claim
            time_diff = now - position.last_reward_claim
            hours_elapsed = Decimal(time_diff.total_seconds()) / Decimal("3600")
            
            if hours_elapsed >= Decimal("24"):  # Daily reward distribution
                # Calculate rewards for the period
                daily_rate = position.apy_rate / Decimal("365")
                rewards = position.amount * daily_rate * (hours_elapsed / Decimal("24"))
                
                # Add to pending rewards
                position.pending_rewards += rewards
                position.last_reward_claim = now
                total_rewards_distributed += rewards
                positions_updated += 1
                
                # Update account
                account = self.accounts[position.account_id]
                account.fog_balance += rewards
                account.total_earned += rewards
        
        # Mint tokens for staking rewards
        if total_rewards_distributed > 0:
            self._mint_tokens(total_rewards_distributed, "staking_rewards")
        
        # Update network utilization (simulate dynamic utilization)
        self._update_network_utilization()
        
        # Record economic metrics
        self._record_economic_metrics()
        
        logger.info(f"Processed periodic rewards: {total_rewards_distributed} FOG distributed "
                   f"to {positions_updated} positions")
        
        return {
            "rewards_distributed": float(total_rewards_distributed),
            "positions_updated": positions_updated,
            "network_utilization": float(self.network_utilization),
            "current_supply": float(self.current_supply),
            "total_staked": float(self.total_staked)
        }
    
    def _calculate_ve_fog(self, amount: Decimal, lock_days: int) -> Decimal:
        """Calculate vote-escrowed FOG based on amount and lock duration."""
        # veFOG = amount * (lock_days / max_lock_days)
        max_lock_days = 730  # 2 years maximum
        multiplier = min(Decimal("1"), Decimal(lock_days) / Decimal(max_lock_days))
        return amount * multiplier
    
    def _calculate_staking_rewards(self, position: StakingPosition) -> Decimal:
        """Calculate accumulated staking rewards for a position."""
        now = datetime.now()
        time_staked = now - position.staked_at
        days_staked = Decimal(time_staked.days)
        
        # Calculate rewards based on APY
        daily_rate = position.apy_rate / Decimal("365")
        rewards = position.amount * daily_rate * days_staked
        
        return rewards + position.pending_rewards
    
    def _calculate_transfer_fee(self, amount: Decimal, token_type: TokenType) -> Decimal:
        """Calculate transfer fee based on amount and token type."""
        if token_type == TokenType.FOG:
            # Progressive fee: 0.01% base + 0.001% per 1000 tokens
            base_fee = amount * Decimal("0.0001")  # 0.01%
            progressive_fee = (amount / Decimal("1000")) * Decimal("0.00001")  # 0.001% per 1K
            return max(self.token_config["base_transaction_fee"], base_fee + progressive_fee)
        else:
            return Decimal("0")  # No fees for other token types
    
    def _mint_tokens(self, amount: Decimal, reason: str):
        """Mint new tokens (increase supply)."""
        if self.current_supply + amount > self.token_config["max_supply"]:
            raise ValueError("Would exceed maximum token supply")
        
        self.current_supply += amount
        
        logger.debug(f"Minted {amount} FOG tokens for {reason}")
    
    def _burn_tokens(self, amount: Decimal, reason: str):
        """Burn tokens (decrease supply)."""
        self.current_supply = max(Decimal("0"), self.current_supply - amount)
        self.total_burned += amount
        
        logger.debug(f"Burned {amount} FOG tokens for {reason}")
    
    def _update_network_utilization(self):
        """Update network utilization based on system activity."""
        # Simulate utilization based on recent activity
        recent_transactions = len([tx for tx in self.transactions 
                                 if tx.timestamp > datetime.now() - timedelta(hours=24)])
        
        # Normalize to 0-1 range (assuming 1000 transactions = 100% utilization)
        base_utilization = min(Decimal("1"), Decimal(recent_transactions) / Decimal("1000"))
        
        # Add some randomness for simulation
        import random
        random_factor = Decimal(str(random.uniform(0.8, 1.2)))
        
        self.network_utilization = max(Decimal("0.1"), 
                                     min(Decimal("1"), base_utilization * random_factor))
    
    def _record_transaction(self, tx_type: TransactionType, token_type: TokenType,
                          from_account: str, to_account: str, amount: Decimal,
                          fee: Decimal = Decimal("0"), memo: str = None,
                          metadata: Dict[str, Any] = None) -> str:
        """Record a transaction in the system."""
        tx_id = f"tx_{uuid.uuid4().hex[:16]}"
        
        transaction = TokenTransaction(
            tx_id=tx_id,
            tx_type=tx_type,
            token_type=token_type,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
            fee=fee,
            timestamp=datetime.now(),
            memo=memo,
            metadata=metadata or {},
            confirmed=True,  # Auto-confirm for simplicity
            confirmation_count=1
        )
        
        self.transactions.append(transaction)
        return tx_id
    
    def _record_economic_metrics(self):
        """Record current economic metrics."""
        active_accounts = len([acc for acc in self.accounts.values() 
                             if acc.last_activity > datetime.now() - timedelta(days=30)])
        
        daily_transactions = len([tx for tx in self.transactions 
                                if tx.timestamp > datetime.now() - timedelta(days=1)])
        
        # Calculate rewards distributed today
        today_rewards = sum(
            tx.amount for tx in self.transactions
            if tx.timestamp > datetime.now() - timedelta(days=1) and
            tx.tx_type in [TransactionType.REWARD, TransactionType.COMPUTE_REWARD, 
                          TransactionType.GOVERNANCE_REWARD]
        )
        
        compute_rewards_today = sum(
            tx.amount for tx in self.transactions
            if tx.timestamp > datetime.now() - timedelta(days=1) and
            tx.tx_type == TransactionType.COMPUTE_REWARD
        )
        
        governance_rewards_today = sum(
            tx.amount for tx in self.transactions
            if tx.timestamp > datetime.now() - timedelta(days=1) and
            tx.tx_type == TransactionType.GOVERNANCE_REWARD
        )
        
        staking_rewards_today = today_rewards - compute_rewards_today - governance_rewards_today
        
        # Calculate participation rates
        total_eligible_accounts = len([acc for acc in self.accounts.values() 
                                     if acc.fog_balance >= self.token_config["min_voting_balance"]])
        
        staking_participation = (self.total_staked / self.current_supply) if self.current_supply > 0 else Decimal("0")
        
        governance_participation = Decimal("0")
        if total_eligible_accounts > 0:
            active_governance_participants = len([acc for acc in self.accounts.values() 
                                                if acc.governance_participation > 0])
            governance_participation = Decimal(active_governance_participants) / Decimal(total_eligible_accounts)
        
        metrics = EconomicMetrics(
            timestamp=datetime.now(),
            total_supply=self.current_supply,
            circulating_supply=self.current_supply - self.total_staked,
            staked_supply=self.total_staked,
            burned_supply=self.total_burned,
            active_accounts=active_accounts,
            daily_transactions=daily_transactions,
            network_utilization=self.network_utilization,
            daily_rewards_distributed=today_rewards,
            compute_rewards=compute_rewards_today,
            staking_rewards=staking_rewards_today,
            governance_rewards=governance_rewards_today,
            reward_sustainability_ratio=today_rewards / max(self.current_supply * self.token_config["inflation_rate"] / 365, Decimal("1")),
            staking_participation_rate=staking_participation,
            governance_participation_rate=governance_participation
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 365 days of metrics
        if len(self.metrics_history) > 365:
            self.metrics_history = self.metrics_history[-365:]
    
    def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        """Get comprehensive account balance and statistics."""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Calculate staking details
        active_positions = [pos for pos in self.staking_positions.values() 
                          if pos.account_id == account_id and pos.active]
        
        pending_rewards = sum(self._calculate_staking_rewards(pos) for pos in active_positions)
        
        # Calculate transaction history
        account_transactions = [
            tx for tx in self.transactions 
            if tx.from_account == account_id or tx.to_account == account_id
        ]
        
        return {
            "account_info": {
                "account_id": account_id,
                "address": account.address,
                "created_at": account.created_at.isoformat(),
                "last_activity": account.last_activity.isoformat(),
                "active": account.active,
                "kyc_verified": account.kyc_verified,
                "tier_level": account.tier_level
            },
            "balances": {
                "fog_balance": float(account.fog_balance),
                "ve_fog_balance": float(account.ve_fog_balance),
                "credit_balance": float(account.credit_balance),
                "compute_balance": float(account.compute_balance),
                "total_staked": float(account.total_staked),
                "pending_rewards": float(pending_rewards)
            },
            "governance": {
                "voting_power": float(account.voting_power),
                "effective_voting_power": float(account.get_effective_voting_power()),
                "governance_participation": account.governance_participation
            },
            "economics": {
                "total_earned": float(account.total_earned),
                "total_spent": float(account.total_spent),
                "compute_contributions": float(account.compute_contributions)
            },
            "staking": {
                "active_positions": len(active_positions),
                "staking_positions": [
                    {
                        "position_id": pos.position_id,
                        "tier": pos.tier.value,
                        "amount": float(pos.amount),
                        "apy": float(pos.apy_rate),
                        "staked_at": pos.staked_at.isoformat(),
                        "unlock_at": pos.unlock_at.isoformat(),
                        "rewards_earned": float(pos.rewards_earned),
                        "pending_rewards": float(self._calculate_staking_rewards(pos))
                    }
                    for pos in active_positions
                ]
            },
            "transaction_summary": {
                "total_transactions": len(account_transactions),
                "last_transaction": account_transactions[-1].timestamp.isoformat() if account_transactions else None
            }
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Calculate token distribution
        balance_distribution = {
            "whales": len([acc for acc in self.accounts.values() if acc.fog_balance >= 1000000]),
            "dolphins": len([acc for acc in self.accounts.values() if 100000 <= acc.fog_balance < 1000000]),
            "fish": len([acc for acc in self.accounts.values() if 10000 <= acc.fog_balance < 100000]),
            "shrimp": len([acc for acc in self.accounts.values() if 1000 <= acc.fog_balance < 10000]),
            "plankton": len([acc for acc in self.accounts.values() if acc.fog_balance < 1000])
        }
        
        # Calculate staking distribution by tier
        staking_by_tier = {}
        for tier in StakingTier:
            tier_positions = [pos for pos in self.staking_positions.values() 
                            if pos.tier == tier and pos.active]
            staking_by_tier[tier.value] = {
                "positions": len(tier_positions),
                "total_amount": float(sum(pos.amount for pos in tier_positions)),
                "avg_amount": float(sum(pos.amount for pos in tier_positions) / max(len(tier_positions), 1))
            }
        
        return {
            "supply_metrics": {
                "total_supply": float(self.current_supply),
                "max_supply": float(self.token_config["max_supply"]),
                "circulating_supply": float(self.current_supply - self.total_staked),
                "staked_supply": float(self.total_staked),
                "burned_supply": float(self.total_burned),
                "inflation_rate": float(self.token_config["inflation_rate"])
            },
            "network_health": {
                "total_accounts": len(self.accounts),
                "active_accounts": len([acc for acc in self.accounts.values() 
                                      if acc.last_activity > datetime.now() - timedelta(days=30)]),
                "verified_accounts": len([acc for acc in self.accounts.values() if acc.kyc_verified]),
                "network_utilization": float(self.network_utilization),
                "daily_transactions": len([tx for tx in self.transactions 
                                         if tx.timestamp > datetime.now() - timedelta(days=1)])
            },
            "staking_metrics": {
                "total_staking_positions": len([pos for pos in self.staking_positions.values() if pos.active]),
                "staking_participation_rate": float(self.total_staked / max(self.current_supply, Decimal("1"))),
                "average_staking_duration": sum(pos.lock_days for pos in self.staking_positions.values() if pos.active) / max(len([pos for pos in self.staking_positions.values() if pos.active]), 1),
                "staking_by_tier": staking_by_tier
            },
            "economic_metrics": {
                "daily_rewards_distributed": float(latest_metrics.daily_rewards_distributed) if latest_metrics else 0,
                "compute_reward_rate": float(self.token_config["base_compute_reward"]),
                "network_utilization_multiplier": float(self.network_utilization * self.token_config["network_utilization_multiplier"]),
                "reward_sustainability_ratio": float(latest_metrics.reward_sustainability_ratio) if latest_metrics else 0
            },
            "distribution": {
                "balance_distribution": balance_distribution,
                "governance_power_distribution": {
                    "total_voting_power": float(sum(acc.voting_power for acc in self.accounts.values())),
                    "average_voting_power": float(sum(acc.voting_power for acc in self.accounts.values()) / max(len(self.accounts), 1))
                }
            },
            "transaction_metrics": {
                "total_transactions": len(self.transactions),
                "avg_daily_transactions": len(self.transactions) / max((datetime.now() - self.accounts[list(self.accounts.keys())[0]].created_at).days, 1) if self.accounts else 0,
                "total_fees_collected": float(sum(tx.fee for tx in self.transactions)),
                "total_fees_burned": float(sum(tx.fee for tx in self.transactions if tx.token_type == TokenType.FOG))
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_transaction_history(self, account_id: str = None, limit: int = 100, 
                               tx_type: TransactionType = None) -> List[Dict[str, Any]]:
        """Get transaction history with optional filtering."""
        transactions = self.transactions
        
        # Filter by account if specified
        if account_id:
            transactions = [tx for tx in transactions 
                          if tx.from_account == account_id or tx.to_account == account_id]
        
        # Filter by transaction type if specified
        if tx_type:
            transactions = [tx for tx in transactions if tx.tx_type == tx_type]
        
        # Sort by timestamp (newest first) and limit
        transactions = sorted(transactions, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "tx_id": tx.tx_id,
                "type": tx.tx_type.value,
                "token_type": tx.token_type.value,
                "from_account": tx.from_account,
                "to_account": tx.to_account,
                "amount": float(tx.amount),
                "fee": float(tx.fee),
                "timestamp": tx.timestamp.isoformat(),
                "confirmed": tx.confirmed,
                "memo": tx.memo,
                "metadata": tx.metadata
            }
            for tx in transactions
        ]
    
    def export_tokenomics_data(self) -> Dict[str, Any]:
        """Export complete tokenomics data for backup or analysis."""
        return {
            "system_config": {
                "token_config": {k: float(v) if isinstance(v, Decimal) else v 
                               for k, v in self.token_config.items()},
                "staking_tiers": {
                    tier.value: {
                        "lock_days": config["lock_days"],
                        "apy": float(config["apy"])
                    }
                    for tier, config in self.staking_tiers.items()
                }
            },
            "current_state": {
                "current_supply": float(self.current_supply),
                "total_staked": float(self.total_staked),
                "total_burned": float(self.total_burned),
                "network_utilization": float(self.network_utilization)
            },
            "accounts": {
                account_id: {
                    **{k: (v.isoformat() if isinstance(v, datetime) else 
                          float(v) if isinstance(v, Decimal) else v)
                       for k, v in account.dict().items()}
                }
                for account_id, account in self.accounts.items()
            },
            "transactions": self.get_transaction_history(limit=10000),
            "staking_positions": {
                pos_id: {
                    **{k: (v.isoformat() if isinstance(v, datetime) else 
                          float(v) if isinstance(v, Decimal) else v)
                       for k, v in position.dict().items()}
                }
                for pos_id, position in self.staking_positions.items()
            },
            "metrics_history": [
                {
                    **{k: (v.isoformat() if isinstance(v, datetime) else 
                          float(v) if isinstance(v, Decimal) else v)
                       for k, v in metrics.dict().items()}
                }
                for metrics in self.metrics_history[-30:]  # Last 30 days
            ],
            "export_timestamp": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of the comprehensive tokenomics system."""
    logger.info("Initializing Comprehensive Tokenomics System...")
    
    # Create tokenomics system
    tokenomics = ComprehensiveTokenomicsSystem()
    
    # Create test user accounts
    alice_id = "alice"
    bob_id = "bob"
    
    tokenomics.create_account(alice_id, "0xalice", Decimal("10000"))
    tokenomics.create_account(bob_id, "0xbob", Decimal("5000"))
    
    # Test token transfers
    tokenomics.transfer_tokens(alice_id, bob_id, Decimal("1000"), memo="Test transfer")
    
    # Test staking
    stake_position = tokenomics.stake_tokens(alice_id, Decimal("5000"), StakingTier.GOLD)
    logger.info(f"Created staking position: {stake_position}")
    
    # Test compute rewards
    compute_reward = tokenomics.distribute_compute_rewards(bob_id, Decimal("10"), Decimal("1.2"))
    logger.info(f"Distributed compute reward: {compute_reward}")
    
    # Test governance rewards
    gov_reward = tokenomics.distribute_governance_rewards(alice_id, 2, 5)
    logger.info(f"Distributed governance reward: {gov_reward}")
    
    # Process periodic rewards
    periodic_results = tokenomics.process_periodic_rewards()
    logger.info(f"Periodic rewards processed: {periodic_results}")
    
    # Get network statistics
    network_stats = tokenomics.get_network_statistics()
    logger.info(f"Network Statistics: {json.dumps(network_stats, indent=2)}")
    
    # Get account balances
    alice_balance = tokenomics.get_account_balance(alice_id)
    logger.info(f"Alice's balance: {json.dumps(alice_balance, indent=2)}")
    
    logger.info("Comprehensive Tokenomics System demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())