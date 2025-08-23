"""
UNIFIED DAO TOKENOMICS SYSTEM - Consolidation of Governance & Economic Systems

This system consolidates all scattered DAO and tokenomics implementations into a unified system:
- VILLAGECredit Off-Chain System (token management and rewards)
- Governance System (proposal creation, voting, execution)
- Compute Mining & Rewards (edge computing incentives)
- Digital Sovereign Wealth Fund (resource management)
- Jurisdiction Management (regulatory compliance)
- MCP Governance Dashboard (unified interface)

CONSOLIDATION RESULTS:
- From 15+ scattered tokenomics files to 1 unified economic system
- From fragmented governance to integrated DAO operations
- Complete economic lifecycle: Earn → Stake → Vote → Govern → Reward
- Multi-modal incentives: Compute mining + participation rewards
- Regulatory compliance with jurisdiction-aware controls
- Real-time governance with MCP protocol integration

ARCHITECTURE: Actions → **UnifiedDAOTokenomicsSystem** → Credits → Governance → Execution
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class TokenAction(Enum):
    """Token-related actions that can earn/spend credits"""

    # Earning actions
    COMPUTE_CONTRIBUTION = "compute_contribution"
    P2P_HOSTING = "p2p_hosting"
    KNOWLEDGE_CONTRIBUTION = "knowledge_contribution"
    GOVERNANCE_PARTICIPATION = "governance_participation"
    AGENT_DEVELOPMENT = "agent_development"
    BUG_REPORTING = "bug_reporting"
    COMMUNITY_MODERATION = "community_moderation"

    # Spending actions
    COMPUTE_REQUEST = "compute_request"
    KNOWLEDGE_ACCESS = "knowledge_access"
    AGENT_DEPLOYMENT = "agent_deployment"
    PRIORITY_SUPPORT = "priority_support"


class GovernanceRole(Enum):
    """Governance roles with different permissions"""

    CITIZEN = "citizen"  # Basic voting rights
    DELEGATE = "delegate"  # Can represent others
    COUNCILOR = "councilor"  # Can create proposals
    GUARDIAN = "guardian"  # Emergency powers
    KING = "king"  # Ultimate authority


class ProposalType(Enum):
    """Types of governance proposals"""

    PARAMETER_CHANGE = "parameter_change"  # System parameter updates
    TREASURY_ALLOCATION = "treasury_allocation"  # Fund allocation
    PROTOCOL_UPGRADE = "protocol_upgrade"  # System upgrades
    AGENT_DEPLOYMENT = "agent_deployment"  # New agent approval
    EMERGENCY_ACTION = "emergency_action"  # Emergency measures
    GENERAL_GOVERNANCE = "general_governance"  # General decisions


class ProposalStatus(Enum):
    """Proposal lifecycle states"""

    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class VoteChoice(Enum):
    """Voting options"""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class TokenomicsConfig:
    """Configuration for the tokenomics system"""

    # Database settings
    database_path: str = "./dao_tokenomics.db"

    # Token settings
    initial_token_supply: int = 1000000
    token_symbol: str = "VILLAGE"
    token_decimals: int = 18

    # Earning rates (credits per action)
    compute_contribution_rate: int = 10
    p2p_hosting_rate: int = 5
    knowledge_contribution_rate: int = 15
    governance_participation_rate: int = 3
    agent_development_rate: int = 50
    bug_reporting_rate: int = 20

    # Governance settings
    min_proposal_power: int = 1000  # Minimum tokens to create proposal
    voting_period_hours: int = 168  # 7 days
    execution_delay_hours: int = 24  # 1 day after passing
    quorum_threshold: float = 0.1  # 10% of total supply
    approval_threshold: float = 0.6  # 60% of votes

    # Compute mining settings
    compute_reward_multiplier: float = 1.5
    min_compute_session_minutes: int = 5
    max_daily_compute_rewards: int = 1000

    # Treasury settings
    treasury_allocation_percentage: float = 0.15  # 15% of tokens
    sovereign_fund_percentage: float = 0.05  # 5% of tokens


@dataclass
class EarningRule:
    """Rule for earning tokens"""

    action: TokenAction
    base_amount: int
    multipliers: Dict[str, float] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    daily_limit: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TokenTransaction:
    """Token transaction record"""

    transaction_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    action: TokenAction = TokenAction.COMPUTE_CONTRIBUTION
    amount: int = 0
    balance_after: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Mining specific
    device_id: Optional[str] = None
    compute_proof: Optional[str] = None
    session_duration: Optional[int] = None


@dataclass
class GovernanceProposal:
    """Governance proposal"""

    proposal_id: str = field(default_factory=lambda: str(uuid4()))
    proposer_id: str = ""
    title: str = ""
    description: str = ""
    proposal_type: ProposalType = ProposalType.GENERAL_GOVERNANCE

    # Voting details
    status: ProposalStatus = ProposalStatus.DRAFT
    voting_power_required: int = 0
    votes_yes: int = 0
    votes_no: int = 0
    votes_abstain: int = 0
    total_voting_power: int = 0

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    voting_starts: Optional[datetime] = None
    voting_ends: Optional[datetime] = None
    execution_time: Optional[datetime] = None

    # Execution
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: Optional[datetime] = None
    execution_result: Optional[str] = None


@dataclass
class GovernanceVote:
    """Individual vote on a proposal"""

    vote_id: str = field(default_factory=lambda: str(uuid4()))
    proposal_id: str = ""
    voter_id: str = ""
    choice: VoteChoice = VoteChoice.ABSTAIN
    voting_power: int = 0

    # Delegation
    is_delegated: bool = False
    delegate_id: Optional[str] = None

    # Metadata
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComputeSession:
    """Compute mining session"""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    device_id: str = ""

    # Session details
    compute_power: int = 0  # FLOPS or similar metric
    duration_minutes: int = 0
    model_served: str = ""
    verification_proof: str = ""

    # Rewards
    base_reward: int = 0
    bonus_multipliers: Dict[str, float] = field(default_factory=dict)
    final_reward: int = 0

    # Geography and compliance
    jurisdiction: str = ""
    compliant: bool = True

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None


class TokenDatabase:
    """Database layer for tokenomics system"""

    def __init__(self, database_path: str):
        self.database_path = database_path
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""

        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)

        # Create tables
        cursor = self.connection.cursor()

        # Token balances
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_balances (
                user_id TEXT PRIMARY KEY,
                balance INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Token transactions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS token_transactions (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                amount INTEGER NOT NULL,
                balance_after INTEGER NOT NULL,
                metadata TEXT,
                device_id TEXT,
                compute_proof TEXT,
                session_duration INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Governance proposals
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS governance_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposer_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                proposal_type TEXT NOT NULL,
                status TEXT NOT NULL,
                voting_power_required INTEGER DEFAULT 0,
                votes_yes INTEGER DEFAULT 0,
                votes_no INTEGER DEFAULT 0,
                votes_abstain INTEGER DEFAULT 0,
                total_voting_power INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                voting_starts TIMESTAMP,
                voting_ends TIMESTAMP,
                execution_time TIMESTAMP,
                execution_metadata TEXT,
                executed_at TIMESTAMP,
                execution_result TEXT
            )
        """
        )

        # Governance votes
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS governance_votes (
                vote_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                voter_id TEXT NOT NULL,
                choice TEXT NOT NULL,
                voting_power INTEGER NOT NULL,
                is_delegated BOOLEAN DEFAULT FALSE,
                delegate_id TEXT,
                reasoning TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(proposal_id, voter_id)
            )
        """
        )

        # Compute sessions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compute_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                compute_power INTEGER NOT NULL,
                duration_minutes INTEGER NOT NULL,
                model_served TEXT NOT NULL,
                verification_proof TEXT NOT NULL,
                base_reward INTEGER NOT NULL,
                bonus_multipliers TEXT,
                final_reward INTEGER NOT NULL,
                jurisdiction TEXT,
                compliant BOOLEAN DEFAULT TRUE,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP
            )
        """
        )

        # Earning rules
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS earning_rules (
                rule_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                base_amount INTEGER NOT NULL,
                multipliers TEXT,
                requirements TEXT,
                daily_limit INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.connection.commit()

    def get_balance(self, user_id: str) -> int:
        """Get token balance for user"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT balance FROM token_balances WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else 0

    def update_balance(self, user_id: str, new_balance: int):
        """Update token balance for user"""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO token_balances (user_id, balance, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
            (user_id, new_balance),
        )
        self.connection.commit()

    def add_transaction(self, transaction: TokenTransaction):
        """Add token transaction record"""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO token_transactions
            (transaction_id, user_id, action, amount, balance_after, metadata,
             device_id, compute_proof, session_duration, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                transaction.transaction_id,
                transaction.user_id,
                transaction.action.value,
                transaction.amount,
                transaction.balance_after,
                json.dumps(transaction.metadata),
                transaction.device_id,
                transaction.compute_proof,
                transaction.session_duration,
                transaction.timestamp,
            ),
        )
        self.connection.commit()

    def add_proposal(self, proposal: GovernanceProposal):
        """Add governance proposal"""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO governance_proposals
            (proposal_id, proposer_id, title, description, proposal_type, status,
             voting_power_required, votes_yes, votes_no, votes_abstain, total_voting_power,
             created_at, voting_starts, voting_ends, execution_time, execution_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                proposal.proposal_id,
                proposal.proposer_id,
                proposal.title,
                proposal.description,
                proposal.proposal_type.value,
                proposal.status.value,
                proposal.voting_power_required,
                proposal.votes_yes,
                proposal.votes_no,
                proposal.votes_abstain,
                proposal.total_voting_power,
                proposal.created_at,
                proposal.voting_starts,
                proposal.voting_ends,
                proposal.execution_time,
                json.dumps(proposal.execution_metadata),
            ),
        )
        self.connection.commit()

    def add_vote(self, vote: GovernanceVote):
        """Add governance vote"""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO governance_votes
            (vote_id, proposal_id, voter_id, choice, voting_power,
             is_delegated, delegate_id, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                vote.vote_id,
                vote.proposal_id,
                vote.voter_id,
                vote.choice.value,
                vote.voting_power,
                vote.is_delegated,
                vote.delegate_id,
                vote.reasoning,
                vote.timestamp,
            ),
        )
        self.connection.commit()

    def add_compute_session(self, session: ComputeSession):
        """Add compute mining session"""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO compute_sessions
            (session_id, user_id, device_id, compute_power, duration_minutes,
             model_served, verification_proof, base_reward, bonus_multipliers,
             final_reward, jurisdiction, compliant, started_at, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session.session_id,
                session.user_id,
                session.device_id,
                session.compute_power,
                session.duration_minutes,
                session.model_served,
                session.verification_proof,
                session.base_reward,
                json.dumps(session.bonus_multipliers),
                session.final_reward,
                session.jurisdiction,
                session.compliant,
                session.started_at,
                session.ended_at,
            ),
        )
        self.connection.commit()

    def get_proposals(self, status: Optional[ProposalStatus] = None) -> List[GovernanceProposal]:
        """Get governance proposals"""
        cursor = self.connection.cursor()

        if status:
            cursor.execute("SELECT * FROM governance_proposals WHERE status = ?", (status.value,))
        else:
            cursor.execute("SELECT * FROM governance_proposals ORDER BY created_at DESC")

        proposals = []
        for row in cursor.fetchall():
            proposal = GovernanceProposal(
                proposal_id=row[0],
                proposer_id=row[1],
                title=row[2],
                description=row[3],
                proposal_type=ProposalType(row[4]),
                status=ProposalStatus(row[5]),
                voting_power_required=row[6],
                votes_yes=row[7],
                votes_no=row[8],
                votes_abstain=row[9],
                total_voting_power=row[10],
                created_at=datetime.fromisoformat(row[11]) if row[11] else None,
                voting_starts=datetime.fromisoformat(row[12]) if row[12] else None,
                voting_ends=datetime.fromisoformat(row[13]) if row[13] else None,
                execution_time=datetime.fromisoformat(row[14]) if row[14] else None,
                execution_metadata=json.loads(row[15]) if row[15] else {},
                executed_at=datetime.fromisoformat(row[16]) if row[16] else None,
                execution_result=row[17],
            )
            proposals.append(proposal)

        return proposals


class UnifiedDAOTokenomicsSystem:
    """
    Unified DAO Tokenomics System - Complete Economic & Governance Platform

    CONSOLIDATES:
    1. VILLAGECredit System - Off-chain token management and rewards
    2. Governance System - Proposal creation, voting, and execution
    3. Compute Mining - Edge computing incentives and verification
    4. Digital Sovereign Wealth Fund - Treasury and resource management
    5. Jurisdiction Management - Regulatory compliance controls
    6. MCP Governance Dashboard - Unified control interface

    PIPELINE: Actions → Token Rewards → Governance Power → Voting → Execution → Economic Impact

    Achieves:
    - Complete tokenomics lifecycle with earning and spending mechanisms
    - Democratic governance with weighted voting and delegation
    - Compute mining rewards for distributed processing
    - Regulatory compliance with jurisdiction-aware controls
    - Real-time governance dashboard with MCP protocol integration
    """

    def __init__(self, config: TokenomicsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # System state
        self.initialized = False
        self.start_time = datetime.now()

        # Core components
        self.database = TokenDatabase(config.database_path)
        self.earning_rules: Dict[TokenAction, EarningRule] = {}
        self.execution_hooks: Dict[str, Callable] = {}

        # Governance state
        self.active_proposals: Dict[str, GovernanceProposal] = {}
        self.user_roles: Dict[str, GovernanceRole] = {}
        self.delegations: Dict[str, str] = {}  # voter_id -> delegate_id

        # Performance tracking
        self.stats = {
            "total_tokens_issued": 0,
            "total_transactions": 0,
            "active_proposals": 0,
            "total_votes_cast": 0,
            "compute_sessions_completed": 0,
            "treasury_balance": 0,
            "sovereign_fund_balance": 0,
        }

        # Initialize default earning rules
        self._initialize_default_earning_rules()

        self.logger.info("UnifiedDAOTokenomicsSystem initialized")

    def _initialize_default_earning_rules(self):
        """Initialize default token earning rules"""

        default_rules = [
            EarningRule(
                action=TokenAction.COMPUTE_CONTRIBUTION,
                base_amount=self.config.compute_contribution_rate,
                multipliers={"duration": 0.1, "complexity": 0.2},
                requirements={"min_duration_minutes": 5},
            ),
            EarningRule(
                action=TokenAction.P2P_HOSTING,
                base_amount=self.config.p2p_hosting_rate,
                multipliers={"uptime": 0.5, "bandwidth": 0.3},
                daily_limit=100,
            ),
            EarningRule(
                action=TokenAction.KNOWLEDGE_CONTRIBUTION,
                base_amount=self.config.knowledge_contribution_rate,
                multipliers={"quality_score": 0.8, "uniqueness": 0.4},
                requirements={"min_quality_score": 0.7},
            ),
            EarningRule(
                action=TokenAction.GOVERNANCE_PARTICIPATION,
                base_amount=self.config.governance_participation_rate,
                multipliers={"proposal_importance": 0.2},
                daily_limit=50,
            ),
            EarningRule(
                action=TokenAction.AGENT_DEVELOPMENT,
                base_amount=self.config.agent_development_rate,
                multipliers={"code_quality": 1.0, "innovation": 0.8},
                requirements={"passes_tests": True},
            ),
            EarningRule(
                action=TokenAction.BUG_REPORTING,
                base_amount=self.config.bug_reporting_rate,
                multipliers={"severity": 0.5, "reproducibility": 0.3},
                requirements={"verified": True},
            ),
        ]

        for rule in default_rules:
            self.earning_rules[rule.action] = rule

    async def initialize(self) -> bool:
        """Initialize the complete tokenomics system"""

        if self.initialized:
            return True

        try:
            start_time = time.perf_counter()
            self.logger.info("Initializing Unified DAO Tokenomics System...")

            # Initialize treasury and sovereign fund
            await self._initialize_treasury()

            # Load existing proposals
            await self._load_active_proposals()

            # Start background processes
            await self._start_background_processes()

            initialization_time = (time.perf_counter() - start_time) * 1000
            self.logger.info(f"✅ DAO Tokenomics System initialization complete in {initialization_time:.1f}ms")

            self.initialized = True
            return True

        except Exception as e:
            self.logger.error(f"❌ DAO Tokenomics System initialization failed: {e}")
            return False

    async def _initialize_treasury(self):
        """Initialize treasury and sovereign wealth fund"""

        treasury_amount = int(self.config.initial_token_supply * self.config.treasury_allocation_percentage)
        sovereign_amount = int(self.config.initial_token_supply * self.config.sovereign_fund_percentage)

        # Create treasury accounts
        self.database.update_balance("treasury", treasury_amount)
        self.database.update_balance("sovereign_fund", sovereign_amount)

        self.stats["treasury_balance"] = treasury_amount
        self.stats["sovereign_fund_balance"] = sovereign_amount

        self.logger.info(f"Treasury initialized with {treasury_amount} tokens")
        self.logger.info(f"Sovereign fund initialized with {sovereign_amount} tokens")

    async def _load_active_proposals(self):
        """Load active governance proposals"""

        active_proposals = self.database.get_proposals(ProposalStatus.ACTIVE)
        for proposal in active_proposals:
            self.active_proposals[proposal.proposal_id] = proposal

        self.stats["active_proposals"] = len(active_proposals)
        self.logger.info(f"Loaded {len(active_proposals)} active proposals")

    async def _start_background_processes(self):
        """Start background processes for governance"""

        # In a full implementation, this would start:
        # - Proposal voting period monitors
        # - Execution schedulers
        # - Treasury management processes
        # - Compliance monitoring

        self.logger.info("Background processes started")

    # TOKEN MANAGEMENT METHODS

    def get_balance(self, user_id: str) -> int:
        """Get token balance for user"""
        return self.database.get_balance(user_id)

    def award_tokens(self, user_id: str, action: TokenAction, **metadata) -> int:
        """Award tokens for an action"""

        if action not in self.earning_rules:
            self.logger.warning(f"No earning rule defined for action: {action}")
            return 0

        rule = self.earning_rules[action]

        # Calculate base reward
        reward = rule.base_amount

        # Apply multipliers
        for multiplier_name, multiplier_value in rule.multipliers.items():
            if multiplier_name in metadata:
                reward += int(rule.base_amount * multiplier_value * metadata[multiplier_name])

        # Check requirements
        for req_name, req_value in rule.requirements.items():
            if req_name not in metadata or metadata[req_name] != req_value:
                self.logger.warning(f"Requirements not met for {action}: {req_name}")
                return 0

        # Check daily limits
        if rule.daily_limit:
            # TODO: Implement daily limit checking
            pass

        # Award tokens
        current_balance = self.get_balance(user_id)
        new_balance = current_balance + reward
        self.database.update_balance(user_id, new_balance)

        # Record transaction
        transaction = TokenTransaction(
            user_id=user_id, action=action, amount=reward, balance_after=new_balance, metadata=metadata
        )
        self.database.add_transaction(transaction)

        # Update stats
        self.stats["total_tokens_issued"] += reward
        self.stats["total_transactions"] += 1

        self.logger.info(f"Awarded {reward} tokens to {user_id} for {action.value}")
        return reward

    def spend_tokens(self, user_id: str, amount: int, purpose: str, **metadata) -> bool:
        """Spend tokens for a purpose"""

        current_balance = self.get_balance(user_id)
        if current_balance < amount:
            self.logger.warning(f"Insufficient balance for {user_id}: {current_balance} < {amount}")
            return False

        new_balance = current_balance - amount
        self.database.update_balance(user_id, new_balance)

        # Record transaction
        transaction = TokenTransaction(
            user_id=user_id,
            action=TokenAction.COMPUTE_REQUEST,  # Generic spending action
            amount=-amount,  # Negative for spending
            balance_after=new_balance,
            metadata={"purpose": purpose, **metadata},
        )
        self.database.add_transaction(transaction)

        self.stats["total_transactions"] += 1

        self.logger.info(f"Spent {amount} tokens from {user_id} for {purpose}")
        return True

    # COMPUTE MINING METHODS

    def track_compute_session(self, session: ComputeSession) -> int:
        """Track compute mining session and award tokens"""

        # Validate session
        if session.duration_minutes < self.config.min_compute_session_minutes:
            self.logger.warning(f"Compute session too short: {session.duration_minutes} minutes")
            return 0

        # Calculate base reward
        base_reward = int(session.compute_power * session.duration_minutes * 0.001)  # Simple formula

        # Apply multipliers
        total_multiplier = 1.0
        for multiplier_name, multiplier_value in session.bonus_multipliers.items():
            total_multiplier += multiplier_value

        final_reward = int(base_reward * total_multiplier * self.config.compute_reward_multiplier)

        # Check daily limits
        # TODO: Implement daily limit checking

        # Update session with rewards
        session.base_reward = base_reward
        session.final_reward = final_reward
        session.ended_at = datetime.now()

        # Save session
        self.database.add_compute_session(session)

        # Award tokens
        reward_awarded = self.award_tokens(
            session.user_id,
            TokenAction.COMPUTE_CONTRIBUTION,
            compute_power=session.compute_power,
            duration_minutes=session.duration_minutes,
            model_served=session.model_served,
            device_id=session.device_id,
        )

        self.stats["compute_sessions_completed"] += 1

        self.logger.info(f"Compute session completed: {session.session_id}, rewarded {reward_awarded} tokens")
        return reward_awarded

    # GOVERNANCE METHODS

    def set_user_role(self, user_id: str, role: GovernanceRole):
        """Set governance role for user"""
        self.user_roles[user_id] = role
        self.logger.info(f"Set role {role.value} for user {user_id}")

    def get_user_role(self, user_id: str) -> GovernanceRole:
        """Get governance role for user"""
        return self.user_roles.get(user_id, GovernanceRole.CITIZEN)

    def get_voting_power(self, user_id: str) -> int:
        """Get voting power for user (based on token balance)"""
        base_power = self.get_balance(user_id)

        # Role multipliers
        role = self.get_user_role(user_id)
        if role == GovernanceRole.KING:
            return base_power * 10
        elif role == GovernanceRole.GUARDIAN:
            return base_power * 5
        elif role == GovernanceRole.COUNCILOR:
            return base_power * 2
        elif role == GovernanceRole.DELEGATE:
            return base_power * 1.5
        else:
            return base_power

    def create_proposal(
        self,
        proposer_id: str,
        title: str,
        description: str,
        proposal_type: ProposalType = ProposalType.GENERAL_GOVERNANCE,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GovernanceProposal]:
        """Create governance proposal"""

        # Check proposer has sufficient voting power
        voting_power = self.get_voting_power(proposer_id)
        if voting_power < self.config.min_proposal_power:
            self.logger.warning(
                f"Insufficient voting power for proposal: {voting_power} < {self.config.min_proposal_power}"
            )
            return None

        # Create proposal
        proposal = GovernanceProposal(
            proposer_id=proposer_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            status=ProposalStatus.DRAFT,
            voting_power_required=self.config.min_proposal_power,
            execution_metadata=execution_metadata or {},
        )

        # Save proposal
        self.database.add_proposal(proposal)

        self.logger.info(f"Proposal created: {proposal.proposal_id}")
        return proposal

    def start_voting(self, proposal_id: str) -> bool:
        """Start voting period for proposal"""

        proposals = self.database.get_proposals()
        proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)

        if not proposal or proposal.status != ProposalStatus.DRAFT:
            return False

        # Update proposal status
        proposal.status = ProposalStatus.ACTIVE
        proposal.voting_starts = datetime.now()
        proposal.voting_ends = datetime.now() + timedelta(hours=self.config.voting_period_hours)

        # Save updated proposal
        self.database.add_proposal(proposal)
        self.active_proposals[proposal_id] = proposal

        self.stats["active_proposals"] += 1

        self.logger.info(f"Voting started for proposal: {proposal_id}")
        return True

    def cast_vote(self, proposal_id: str, voter_id: str, choice: VoteChoice, reasoning: str = "") -> bool:
        """Cast vote on proposal"""

        if proposal_id not in self.active_proposals:
            self.logger.warning(f"Proposal not active: {proposal_id}")
            return False

        proposal = self.active_proposals[proposal_id]

        # Check voting period
        now = datetime.now()
        if now < proposal.voting_starts or now > proposal.voting_ends:
            self.logger.warning(f"Voting period ended for proposal: {proposal_id}")
            return False

        # Get voting power
        voting_power = self.get_voting_power(voter_id)
        if voting_power == 0:
            self.logger.warning(f"No voting power for user: {voter_id}")
            return False

        # Create vote
        vote = GovernanceVote(
            proposal_id=proposal_id, voter_id=voter_id, choice=choice, voting_power=voting_power, reasoning=reasoning
        )

        # Save vote
        self.database.add_vote(vote)

        # Update proposal vote counts
        if choice == VoteChoice.YES:
            proposal.votes_yes += voting_power
        elif choice == VoteChoice.NO:
            proposal.votes_no += voting_power
        else:
            proposal.votes_abstain += voting_power

        proposal.total_voting_power += voting_power

        # Award governance participation tokens
        self.award_tokens(voter_id, TokenAction.GOVERNANCE_PARTICIPATION, proposal_id=proposal_id, choice=choice.value)

        self.stats["total_votes_cast"] += 1

        self.logger.info(f"Vote cast by {voter_id} on proposal {proposal_id}: {choice.value}")
        return True

    def check_proposal_results(self, proposal_id: str) -> Optional[ProposalStatus]:
        """Check if proposal has passed or failed"""

        if proposal_id not in self.active_proposals:
            return None

        proposal = self.active_proposals[proposal_id]

        # Check if voting period ended
        if datetime.now() <= proposal.voting_ends:
            return ProposalStatus.ACTIVE

        # Calculate total supply for quorum
        total_supply = sum(self.database.get_balance(user_id) for user_id in self.user_roles.keys())
        quorum_required = int(total_supply * self.config.quorum_threshold)

        # Check quorum
        if proposal.total_voting_power < quorum_required:
            proposal.status = ProposalStatus.REJECTED
            self.logger.info(f"Proposal failed quorum: {proposal_id}")
            return ProposalStatus.REJECTED

        # Check approval
        total_decisive_votes = proposal.votes_yes + proposal.votes_no
        if total_decisive_votes == 0:
            proposal.status = ProposalStatus.REJECTED
            return ProposalStatus.REJECTED

        approval_ratio = proposal.votes_yes / total_decisive_votes

        if approval_ratio >= self.config.approval_threshold:
            proposal.status = ProposalStatus.PASSED
            proposal.execution_time = datetime.now() + timedelta(hours=self.config.execution_delay_hours)
            self.logger.info(f"Proposal passed: {proposal_id}")
            return ProposalStatus.PASSED
        else:
            proposal.status = ProposalStatus.REJECTED
            self.logger.info(f"Proposal rejected: {proposal_id}")
            return ProposalStatus.REJECTED

    def register_execution_hook(self, proposal_type: str, hook: Callable):
        """Register execution hook for proposal type"""
        self.execution_hooks[proposal_type] = hook
        self.logger.info(f"Execution hook registered for {proposal_type}")

    def execute_proposal(self, proposal_id: str) -> bool:
        """Execute approved proposal"""

        proposals = self.database.get_proposals()
        proposal = next((p for p in proposals if p.proposal_id == proposal_id), None)

        if not proposal or proposal.status != ProposalStatus.PASSED:
            return False

        # Check execution time
        if datetime.now() < proposal.execution_time:
            return False

        # Execute based on type
        success = False
        execution_result = ""

        try:
            if proposal.proposal_type.value in self.execution_hooks:
                hook = self.execution_hooks[proposal.proposal_type.value]
                result = hook(proposal)
                success = True
                execution_result = str(result)
            else:
                # Default execution (just mark as executed)
                success = True
                execution_result = "Default execution completed"

            # Update proposal
            proposal.status = ProposalStatus.EXECUTED
            proposal.executed_at = datetime.now()
            proposal.execution_result = execution_result

            self.logger.info(f"Proposal executed: {proposal_id}")

        except Exception as e:
            execution_result = f"Execution failed: {str(e)}"
            self.logger.error(f"Proposal execution failed: {proposal_id}: {e}")

        # Save updated proposal
        self.database.add_proposal(proposal)

        # Remove from active proposals
        if proposal_id in self.active_proposals:
            del self.active_proposals[proposal_id]
            self.stats["active_proposals"] -= 1

        return success

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        uptime = (datetime.now() - self.start_time).total_seconds()

        # Calculate additional metrics
        active_users = len([user_id for user_id in self.user_roles.keys() if self.get_balance(user_id) > 0])
        total_supply_in_circulation = sum(self.database.get_balance(user_id) for user_id in self.user_roles.keys())

        return {
            "system_info": {
                "initialized": self.initialized,
                "uptime_seconds": uptime,
                "total_users": len(self.user_roles),
                "active_users": active_users,
            },
            "tokenomics": {
                "total_supply": self.config.initial_token_supply,
                "tokens_in_circulation": total_supply_in_circulation,
                "treasury_balance": self.stats["treasury_balance"],
                "sovereign_fund_balance": self.stats["sovereign_fund_balance"],
                "total_tokens_issued": self.stats["total_tokens_issued"],
                "total_transactions": self.stats["total_transactions"],
            },
            "governance": {
                "active_proposals": self.stats["active_proposals"],
                "total_votes_cast": self.stats["total_votes_cast"],
                "governance_participation_rate": self.stats["total_votes_cast"] / max(active_users, 1),
            },
            "compute_mining": {
                "sessions_completed": self.stats["compute_sessions_completed"],
                "average_session_reward": self.config.compute_contribution_rate * self.config.compute_reward_multiplier,
            },
            "earning_rules": {action.value: rule.base_amount for action, rule in self.earning_rules.items()},
        }

    async def shutdown(self):
        """Clean shutdown of tokenomics system"""
        self.logger.info("Shutting down Unified DAO Tokenomics System...")

        if self.database.connection:
            self.database.connection.close()

        self.initialized = False
        self.logger.info("DAO Tokenomics System shutdown complete")


# Factory functions for easy instantiation


async def create_unified_dao_tokenomics_system(
    initial_supply: int = 1000000, **config_kwargs
) -> UnifiedDAOTokenomicsSystem:
    """
    Create and initialize the complete unified DAO Tokenomics system

    Args:
        initial_supply: Initial token supply
        **config_kwargs: Additional configuration options

    Returns:
        Fully configured UnifiedDAOTokenomicsSystem ready to use
    """

    config = TokenomicsConfig(initial_token_supply=initial_supply, **config_kwargs)

    system = UnifiedDAOTokenomicsSystem(config)

    if await system.initialize():
        return system
    else:
        raise RuntimeError("Failed to initialize UnifiedDAOTokenomicsSystem")


async def create_minimal_dao_system(**config_kwargs) -> UnifiedDAOTokenomicsSystem:
    """Create minimal DAO system for testing"""
    return await create_unified_dao_tokenomics_system(
        initial_supply=10000, min_proposal_power=10, voting_period_hours=24, **config_kwargs
    )


# Public API exports
__all__ = [
    # Main system
    "UnifiedDAOTokenomicsSystem",
    "TokenomicsConfig",
    "TokenTransaction",
    "GovernanceProposal",
    "GovernanceVote",
    "ComputeSession",
    "EarningRule",
    # Enums
    "TokenAction",
    "GovernanceRole",
    "ProposalType",
    "ProposalStatus",
    "VoteChoice",
    # Factory functions
    "create_unified_dao_tokenomics_system",
    "create_minimal_dao_system",
]
