"""
DAO Governance Operational System

Complete operational procedures for the AIVillage DAO including:
- Enhanced voting system with delegate and quorum management
- Proposal lifecycle management with review processes
- Member management and role-based permissions
- Integration with existing tokenomics and monitoring
- Automated governance reporting and compliance
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from ...shared.compliance.pii_phi_manager import PIIPHIManager
from ..monitoring.slo_monitor import SLOMonitor
from ..tokenomics.fog_token_system import FogTokenSystem

logger = logging.getLogger(__name__)


class MemberRole(Enum):
    """DAO member roles with different permissions."""

    CITIZEN = "citizen"  # Basic voting rights
    DELEGATE = "delegate"  # Can receive delegated votes
    COUNCIL = "council"  # Can fast-track proposals
    ADMIN = "admin"  # System administration
    FOUNDER = "founder"  # Full governance rights


class ProposalCategory(Enum):
    """Categories of governance proposals."""

    ECONOMIC = "economic"  # Tokenomics, rewards, treasury
    TECHNICAL = "technical"  # System upgrades, features
    GOVERNANCE = "governance"  # DAO structure, voting rules
    COMPLIANCE = "compliance"  # Legal, regulatory changes
    COMMUNITY = "community"  # Community initiatives
    EMERGENCY = "emergency"  # Critical system issues


class ProposalPhase(Enum):
    """Phases in proposal lifecycle."""

    DRAFT = "draft"  # Being prepared
    REVIEW = "review"  # Under review by council
    DISCUSSION = "discussion"  # Public discussion period
    VOTING = "voting"  # Active voting
    EXECUTION = "execution"  # Being implemented
    COMPLETED = "completed"  # Successfully completed
    REJECTED = "rejected"  # Failed to pass
    CANCELLED = "cancelled"  # Cancelled by proposer


@dataclass
class DAOMember:
    """DAO member with governance rights."""

    member_id: str
    wallet_address: str
    role: MemberRole

    # Voting power and delegation
    voting_power: int = 0
    delegated_power: int = 0
    delegates_to: str | None = None
    delegators: set[str] = field(default_factory=set)

    # Participation metrics
    proposals_created: int = 0
    votes_cast: int = 0
    participation_rate: float = 0.0
    reputation_score: int = 100

    # Membership details
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False

    # Permissions
    can_create_proposals: bool = True
    can_vote: bool = True
    can_delegate: bool = True


@dataclass
class GovernanceProposal:
    """Enhanced DAO governance proposal."""

    proposal_id: str
    title: str
    description: str
    category: ProposalCategory
    proposer_id: str

    # Proposal content
    summary: str = ""
    detailed_specification: str = ""
    implementation_plan: str = ""
    success_criteria: str = ""
    risk_assessment: str = ""

    # Lifecycle
    phase: ProposalPhase = ProposalPhase.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    review_deadline: datetime | None = None
    discussion_deadline: datetime | None = None
    voting_start: datetime | None = None
    voting_end: datetime | None = None

    # Voting configuration
    quorum_required: int = 0
    approval_threshold: float = 0.5  # 50% by default
    voting_type: str = "simple"  # simple, ranked, quadratic

    # Review and approval
    reviewer_assigned: str | None = None
    review_comments: list[str] = field(default_factory=list)
    council_approved: bool = False

    # Voting results
    total_votes: int = 0
    yes_votes: int = 0
    no_votes: int = 0
    abstain_votes: int = 0
    participant_count: int = 0

    # Execution
    execution_status: str = "pending"
    execution_results: dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class VotingRecord:
    """Record of a member's vote on a proposal."""

    vote_id: str
    proposal_id: str
    member_id: str
    choice: str  # yes, no, abstain
    voting_power: int
    timestamp: datetime
    delegated: bool = False
    delegate_id: str | None = None


@dataclass
class DelegationRecord:
    """Record of vote delegation between members."""

    delegation_id: str
    delegator_id: str
    delegate_id: str
    voting_power: int
    created_at: datetime
    expires_at: datetime | None = None
    active: bool = True


class DAOOperationalSystem:
    """
    Complete DAO operational system for AIVillage governance.

    Provides:
    - Member management with roles and permissions
    - Enhanced proposal lifecycle with review processes
    - Voting system with delegation and quorum management
    - Integration with tokenomics and compliance systems
    - Automated reporting and monitoring
    """

    def __init__(
        self,
        token_system: FogTokenSystem,
        compliance_manager: PIIPHIManager,
        slo_monitor: SLOMonitor,
        data_dir: str = "governance_data",
    ):
        self.token_system = token_system
        self.compliance_manager = compliance_manager
        self.slo_monitor = slo_monitor

        # Data storage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "dao_governance.db"

        # In-memory state
        self.members: dict[str, DAOMember] = {}
        self.proposals: dict[str, GovernanceProposal] = {}
        self.voting_records: list[VotingRecord] = []
        self.delegations: dict[str, DelegationRecord] = {}

        # Configuration
        self.config = self._load_config()

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        # Initialize database
        self._init_database()
        self._load_existing_data()

        logger.info("DAO Operational System initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load DAO configuration."""
        return {
            "governance": {
                "min_proposal_stake": 1000,  # FOG tokens required to create proposal
                "discussion_period_hours": 72,  # 3 days
                "voting_period_hours": 168,  # 7 days
                "quorum_percentage": 10,  # 10% of total voting power
                "approval_threshold": 0.5,  # 50% approval required
                "emergency_quorum_percentage": 5,  # Emergency proposals
                "council_fast_track_threshold": 0.8,  # Council fast-track threshold
                "reputation_voting_multiplier": 1.5,  # Reputation bonus
            },
            "membership": {
                "citizen_min_tokens": 100,  # Minimum tokens for citizenship
                "delegate_min_tokens": 5000,  # Minimum tokens to be delegate
                "council_min_tokens": 50000,  # Minimum tokens for council
                "verification_required": True,
                "auto_delegation_timeout_days": 30,
            },
            "compliance": {
                "proposal_compliance_check": True,
                "voting_privacy_protection": True,
                "audit_trail_retention_days": 2555,  # 7 years
                "reporting_schedule": "weekly",
            },
        }

    def _init_database(self):
        """Initialize governance database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Members table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS members (
                member_id TEXT PRIMARY KEY,
                wallet_address TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                voting_power INTEGER DEFAULT 0,
                delegated_power INTEGER DEFAULT 0,
                delegates_to TEXT,
                proposals_created INTEGER DEFAULT 0,
                votes_cast INTEGER DEFAULT 0,
                participation_rate REAL DEFAULT 0.0,
                reputation_score INTEGER DEFAULT 100,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified BOOLEAN DEFAULT FALSE,
                can_create_proposals BOOLEAN DEFAULT TRUE,
                can_vote BOOLEAN DEFAULT TRUE,
                can_delegate BOOLEAN DEFAULT TRUE,
                delegators TEXT DEFAULT '[]'  -- JSON array
            )
        """
        )

        # Proposals table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                proposer_id TEXT NOT NULL,
                summary TEXT,
                detailed_specification TEXT,
                implementation_plan TEXT,
                success_criteria TEXT,
                risk_assessment TEXT,
                phase TEXT DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                review_deadline TIMESTAMP,
                discussion_deadline TIMESTAMP,
                voting_start TIMESTAMP,
                voting_end TIMESTAMP,
                quorum_required INTEGER DEFAULT 0,
                approval_threshold REAL DEFAULT 0.5,
                voting_type TEXT DEFAULT 'simple',
                reviewer_assigned TEXT,
                review_comments TEXT DEFAULT '[]',  -- JSON array
                council_approved BOOLEAN DEFAULT FALSE,
                total_votes INTEGER DEFAULT 0,
                yes_votes INTEGER DEFAULT 0,
                no_votes INTEGER DEFAULT 0,
                abstain_votes INTEGER DEFAULT 0,
                participant_count INTEGER DEFAULT 0,
                execution_status TEXT DEFAULT 'pending',
                execution_results TEXT DEFAULT '{}',  -- JSON object
                tags TEXT DEFAULT '[]',  -- JSON array
                attachments TEXT DEFAULT '[]',  -- JSON array
                dependencies TEXT DEFAULT '[]',  -- JSON array
                FOREIGN KEY (proposer_id) REFERENCES members(member_id)
            )
        """
        )

        # Voting records table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS voting_records (
                vote_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                member_id TEXT NOT NULL,
                choice TEXT NOT NULL,
                voting_power INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                delegated BOOLEAN DEFAULT FALSE,
                delegate_id TEXT,
                FOREIGN KEY (proposal_id) REFERENCES proposals(proposal_id),
                FOREIGN KEY (member_id) REFERENCES members(member_id),
                FOREIGN KEY (delegate_id) REFERENCES members(member_id)
            )
        """
        )

        # Delegations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS delegations (
                delegation_id TEXT PRIMARY KEY,
                delegator_id TEXT NOT NULL,
                delegate_id TEXT NOT NULL,
                voting_power INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (delegator_id) REFERENCES members(member_id),
                FOREIGN KEY (delegate_id) REFERENCES members(member_id)
            )
        """
        )

        # Audit log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS governance_audit_log (
                audit_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                actor_id TEXT,
                target_id TEXT,
                details TEXT,  -- JSON object
                compliance_checked BOOLEAN DEFAULT FALSE,
                risk_level TEXT DEFAULT 'low'
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_role ON members(role)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_voting_power ON members(voting_power)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_phase ON proposals(phase)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_category ON proposals(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_voting_proposal ON voting_records(proposal_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_voting_member ON voting_records(member_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_delegations_active ON delegations(active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON governance_audit_log(timestamp)")

        conn.commit()
        conn.close()

        logger.info("Governance database initialized")

    def _load_existing_data(self):
        """Load existing governance data from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load members
        cursor.execute("SELECT * FROM members")
        for row in cursor.fetchall():
            member = DAOMember(
                member_id=row["member_id"],
                wallet_address=row["wallet_address"],
                role=MemberRole(row["role"]),
                voting_power=row["voting_power"],
                delegated_power=row["delegated_power"],
                delegates_to=row["delegates_to"],
                proposals_created=row["proposals_created"],
                votes_cast=row["votes_cast"],
                participation_rate=row["participation_rate"],
                reputation_score=row["reputation_score"],
                joined_at=datetime.fromisoformat(row["joined_at"]),
                last_active=datetime.fromisoformat(row["last_active"]),
                verified=bool(row["verified"]),
                can_create_proposals=bool(row["can_create_proposals"]),
                can_vote=bool(row["can_vote"]),
                can_delegate=bool(row["can_delegate"]),
                delegators=set(json.loads(row["delegators"])),
            )
            self.members[member.member_id] = member

        # Load proposals
        cursor.execute("SELECT * FROM proposals")
        for row in cursor.fetchall():
            proposal = GovernanceProposal(
                proposal_id=row["proposal_id"],
                title=row["title"],
                description=row["description"],
                category=ProposalCategory(row["category"]),
                proposer_id=row["proposer_id"],
                summary=row["summary"] or "",
                detailed_specification=row["detailed_specification"] or "",
                implementation_plan=row["implementation_plan"] or "",
                success_criteria=row["success_criteria"] or "",
                risk_assessment=row["risk_assessment"] or "",
                phase=ProposalPhase(row["phase"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                review_deadline=datetime.fromisoformat(row["review_deadline"]) if row["review_deadline"] else None,
                discussion_deadline=datetime.fromisoformat(row["discussion_deadline"])
                if row["discussion_deadline"]
                else None,
                voting_start=datetime.fromisoformat(row["voting_start"]) if row["voting_start"] else None,
                voting_end=datetime.fromisoformat(row["voting_end"]) if row["voting_end"] else None,
                quorum_required=row["quorum_required"],
                approval_threshold=row["approval_threshold"],
                voting_type=row["voting_type"],
                reviewer_assigned=row["reviewer_assigned"],
                review_comments=json.loads(row["review_comments"]),
                council_approved=bool(row["council_approved"]),
                total_votes=row["total_votes"],
                yes_votes=row["yes_votes"],
                no_votes=row["no_votes"],
                abstain_votes=row["abstain_votes"],
                participant_count=row["participant_count"],
                execution_status=row["execution_status"],
                execution_results=json.loads(row["execution_results"]),
                tags=json.loads(row["tags"]),
                attachments=json.loads(row["attachments"]),
                dependencies=json.loads(row["dependencies"]),
            )
            self.proposals[proposal.proposal_id] = proposal

        # Load delegations
        cursor.execute("SELECT * FROM delegations WHERE active = TRUE")
        for row in cursor.fetchall():
            delegation = DelegationRecord(
                delegation_id=row["delegation_id"],
                delegator_id=row["delegator_id"],
                delegate_id=row["delegate_id"],
                voting_power=row["voting_power"],
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                active=bool(row["active"]),
            )
            self.delegations[delegation.delegation_id] = delegation

        conn.close()

        logger.info(
            f"Loaded {len(self.members)} members, {len(self.proposals)} proposals, {len(self.delegations)} delegations"
        )

    async def start(self):
        """Start the DAO operational system."""
        if self._running:
            return

        logger.info("Starting DAO Operational System")
        self._running = True

        # Start background tasks
        tasks = [
            self._proposal_lifecycle_manager(),
            self._voting_monitor(),
            self._delegation_manager(),
            self._compliance_monitor(),
            self._governance_reporter(),
            self._reputation_updater(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("DAO Operational System started")

    async def stop(self):
        """Stop the DAO operational system."""
        if not self._running:
            return

        logger.info("Stopping DAO Operational System")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("DAO Operational System stopped")

    # Member Management

    async def register_member(
        self, wallet_address: str, role: MemberRole = MemberRole.CITIZEN, verification_required: bool = True
    ) -> str:
        """Register a new DAO member."""

        # Check if member already exists
        for member in self.members.values():
            if member.wallet_address == wallet_address:
                raise ValueError(f"Member with wallet {wallet_address} already registered")

        # Get member's token balance
        account_info = self.token_system.get_account_balance(wallet_address)
        if "error" in account_info:
            raise ValueError(f"Wallet {wallet_address} not found in token system")

        token_balance = account_info["balance"]

        # Check minimum token requirements
        min_tokens = self.config["membership"]["citizen_min_tokens"]
        if role == MemberRole.DELEGATE:
            min_tokens = self.config["membership"]["delegate_min_tokens"]
        elif role == MemberRole.COUNCIL:
            min_tokens = self.config["membership"]["council_min_tokens"]

        if token_balance < min_tokens:
            raise ValueError(
                f"Insufficient tokens for {role.value} role. Required: {min_tokens}, Available: {token_balance}"
            )

        # Create member
        member_id = str(uuid.uuid4())
        member = DAOMember(
            member_id=member_id,
            wallet_address=wallet_address,
            role=role,
            voting_power=int(token_balance),
            verified=not verification_required,
        )

        self.members[member_id] = member
        await self._save_member(member)

        # Log registration
        await self._log_governance_event(
            "member_registered",
            actor_id=member_id,
            details={"wallet_address": wallet_address, "role": role.value, "voting_power": member.voting_power},
        )

        logger.info(f"Registered new member: {member_id} ({role.value}) with {token_balance} voting power")
        return member_id

    async def update_member_voting_power(self, member_id: str) -> int:
        """Update member's voting power based on current token balance."""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")

        member = self.members[member_id]
        account_info = self.token_system.get_account_balance(member.wallet_address)

        if "error" not in account_info:
            old_power = member.voting_power
            member.voting_power = int(account_info["balance"])
            member.last_active = datetime.utcnow()

            await self._save_member(member)

            if old_power != member.voting_power:
                await self._log_governance_event(
                    "voting_power_updated",
                    actor_id=member_id,
                    details={"old_power": old_power, "new_power": member.voting_power},
                )

        return member.voting_power

    async def delegate_voting_power(self, delegator_id: str, delegate_id: str, expires_hours: int | None = None) -> str:
        """Delegate voting power to another member."""

        if delegator_id not in self.members or delegate_id not in self.members:
            raise ValueError("Invalid member IDs")

        delegator = self.members[delegator_id]
        delegate = self.members[delegate_id]

        # Check permissions
        if not delegator.can_delegate:
            raise ValueError("Member cannot delegate voting power")

        if delegate.role not in [MemberRole.DELEGATE, MemberRole.COUNCIL, MemberRole.ADMIN]:
            raise ValueError("Target member cannot receive delegations")

        # Update member states
        await self.update_member_voting_power(delegator_id)
        await self.update_member_voting_power(delegate_id)

        # Create delegation record
        delegation_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours) if expires_hours else None

        delegation = DelegationRecord(
            delegation_id=delegation_id,
            delegator_id=delegator_id,
            delegate_id=delegate_id,
            voting_power=delegator.voting_power,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
        )

        # Update member states
        delegator.delegates_to = delegate_id
        delegate.delegated_power += delegator.voting_power
        delegate.delegators.add(delegator_id)

        self.delegations[delegation_id] = delegation
        await self._save_delegation(delegation)
        await self._save_member(delegator)
        await self._save_member(delegate)

        await self._log_governance_event(
            "voting_power_delegated",
            actor_id=delegator_id,
            target_id=delegate_id,
            details={
                "voting_power": delegator.voting_power,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )

        logger.info(f"Delegated {delegator.voting_power} voting power from {delegator_id} to {delegate_id}")
        return delegation_id

    async def revoke_delegation(self, delegator_id: str) -> bool:
        """Revoke a member's delegation."""
        if delegator_id not in self.members:
            raise ValueError(f"Member {delegator_id} not found")

        delegator = self.members[delegator_id]
        if not delegator.delegates_to:
            return False

        delegate_id = delegator.delegates_to
        delegate = self.members[delegate_id]

        # Find and deactivate delegation
        for delegation in self.delegations.values():
            if delegation.delegator_id == delegator_id and delegation.active:
                delegation.active = False
                await self._save_delegation(delegation)
                break

        # Update member states
        delegate.delegated_power -= delegator.voting_power
        delegate.delegators.discard(delegator_id)
        delegator.delegates_to = None

        await self._save_member(delegator)
        await self._save_member(delegate)

        await self._log_governance_event(
            "delegation_revoked",
            actor_id=delegator_id,
            target_id=delegate_id,
            details={"voting_power": delegator.voting_power},
        )

        logger.info(f"Revoked delegation from {delegator_id} to {delegate_id}")
        return True

    # Proposal Management

    async def create_proposal(
        self, proposer_id: str, title: str, description: str, category: ProposalCategory, **kwargs
    ) -> str:
        """Create a new governance proposal."""

        if proposer_id not in self.members:
            raise ValueError(f"Member {proposer_id} not found")

        member = self.members[proposer_id]

        # Check permissions
        if not member.can_create_proposals:
            raise ValueError("Member cannot create proposals")

        # Check minimum stake requirement
        min_stake = self.config["governance"]["min_proposal_stake"]
        if member.voting_power < min_stake:
            raise ValueError(f"Insufficient voting power to create proposal. Required: {min_stake}")

        # Compliance check
        if self.config["compliance"]["proposal_compliance_check"]:
            compliance_result = await self.compliance_manager.scan_job_inputs_for_compliance(
                job_id=f"proposal_scan_{int(time.time())}",
                job_inputs={"title": title, "description": description, "metadata": kwargs},
            )

            if compliance_result["risk_level"] == "HIGH":
                raise ValueError("Proposal failed compliance check: " + str(compliance_result["violations"]))

        # Create proposal
        proposal_id = str(uuid.uuid4())
        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            category=category,
            proposer_id=proposer_id,
            summary=kwargs.get("summary", ""),
            detailed_specification=kwargs.get("specification", ""),
            implementation_plan=kwargs.get("implementation_plan", ""),
            success_criteria=kwargs.get("success_criteria", ""),
            risk_assessment=kwargs.get("risk_assessment", ""),
            tags=kwargs.get("tags", []),
            dependencies=kwargs.get("dependencies", []),
        )

        # Set review deadline
        proposal.review_deadline = datetime.utcnow() + timedelta(hours=24)

        # Calculate quorum requirement
        total_voting_power = sum(m.voting_power + m.delegated_power for m in self.members.values())
        quorum_percentage = self.config["governance"]["quorum_percentage"]
        if category == ProposalCategory.EMERGENCY:
            quorum_percentage = self.config["governance"]["emergency_quorum_percentage"]

        proposal.quorum_required = int(total_voting_power * quorum_percentage / 100)

        self.proposals[proposal_id] = proposal
        await self._save_proposal(proposal)

        # Update member stats
        member.proposals_created += 1
        await self._save_member(member)

        await self._log_governance_event(
            "proposal_created",
            actor_id=proposer_id,
            target_id=proposal_id,
            details={"title": title, "category": category.value, "quorum_required": proposal.quorum_required},
        )

        logger.info(f"Created proposal: {proposal_id} ({title}) by {proposer_id}")
        return proposal_id

    async def advance_proposal_phase(self, proposal_id: str, force: bool = False) -> ProposalPhase:
        """Advance proposal to next lifecycle phase."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]
        current_phase = proposal.phase

        if current_phase == ProposalPhase.DRAFT:
            # Move to review
            proposal.phase = ProposalPhase.REVIEW
            proposal.review_deadline = datetime.utcnow() + timedelta(hours=24)

            # Assign reviewer (council member with least active reviews)
            council_members = [m for m in self.members.values() if m.role == MemberRole.COUNCIL]
            if council_members:
                # Simple assignment - in practice would be more sophisticated
                proposal.reviewer_assigned = council_members[0].member_id

        elif current_phase == ProposalPhase.REVIEW:
            if not proposal.council_approved and not force:
                raise ValueError("Proposal must be approved by council before advancing")

            # Move to discussion
            proposal.phase = ProposalPhase.DISCUSSION
            discussion_hours = self.config["governance"]["discussion_period_hours"]
            proposal.discussion_deadline = datetime.utcnow() + timedelta(hours=discussion_hours)

        elif current_phase == ProposalPhase.DISCUSSION:
            # Move to voting
            proposal.phase = ProposalPhase.VOTING
            voting_hours = self.config["governance"]["voting_period_hours"]
            proposal.voting_start = datetime.utcnow()
            proposal.voting_end = proposal.voting_start + timedelta(hours=voting_hours)

        elif current_phase == ProposalPhase.VOTING:
            # Determine outcome and move to execution or completion
            if proposal.total_votes >= proposal.quorum_required:
                approval_rate = proposal.yes_votes / proposal.total_votes if proposal.total_votes > 0 else 0
                if approval_rate >= proposal.approval_threshold:
                    proposal.phase = ProposalPhase.EXECUTION
                    proposal.execution_status = "approved"
                else:
                    proposal.phase = ProposalPhase.REJECTED
            else:
                proposal.phase = ProposalPhase.REJECTED

        elif current_phase == ProposalPhase.EXECUTION:
            # Mark as completed
            proposal.phase = ProposalPhase.COMPLETED
            proposal.execution_status = "completed"

        await self._save_proposal(proposal)

        await self._log_governance_event(
            "proposal_phase_advanced",
            target_id=proposal_id,
            details={"from_phase": current_phase.value, "to_phase": proposal.phase.value},
        )

        logger.info(f"Advanced proposal {proposal_id} from {current_phase.value} to {proposal.phase.value}")
        return proposal.phase

    async def cast_vote(
        self, proposal_id: str, member_id: str, choice: str, use_delegation: bool = True  # "yes", "no", "abstain"
    ) -> str:
        """Cast a vote on a proposal."""

        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")

        proposal = self.proposals[proposal_id]
        member = self.members[member_id]

        # Check voting phase
        if proposal.phase != ProposalPhase.VOTING:
            raise ValueError(f"Proposal is not in voting phase (current: {proposal.phase.value})")

        # Check voting period
        current_time = datetime.utcnow()
        if current_time < proposal.voting_start or current_time > proposal.voting_end:
            raise ValueError("Voting period is not active")

        # Check permissions
        if not member.can_vote:
            raise ValueError("Member cannot vote")

        # Update member voting power
        await self.update_member_voting_power(member_id)

        # Calculate total voting power (own + delegated)
        total_voting_power = member.voting_power
        if use_delegation:
            total_voting_power += member.delegated_power

        if total_voting_power <= 0:
            raise ValueError("Member has no voting power")

        # Check if member already voted
        existing_vote = None
        for vote_record in self.voting_records:
            if vote_record.proposal_id == proposal_id and vote_record.member_id == member_id:
                existing_vote = vote_record
                break

        vote_id = str(uuid.uuid4())

        # Create or update vote record
        vote_record = VotingRecord(
            vote_id=vote_id,
            proposal_id=proposal_id,
            member_id=member_id,
            choice=choice,
            voting_power=total_voting_power,
            timestamp=current_time,
        )

        # Update proposal vote counts
        if existing_vote:
            # Remove old vote counts
            if existing_vote.choice == "yes":
                proposal.yes_votes -= existing_vote.voting_power
            elif existing_vote.choice == "no":
                proposal.no_votes -= existing_vote.voting_power
            elif existing_vote.choice == "abstain":
                proposal.abstain_votes -= existing_vote.voting_power
            proposal.total_votes -= existing_vote.voting_power

            # Remove old record
            self.voting_records = [v for v in self.voting_records if v.vote_id != existing_vote.vote_id]
        else:
            proposal.participant_count += 1

        # Add new vote counts
        if choice == "yes":
            proposal.yes_votes += total_voting_power
        elif choice == "no":
            proposal.no_votes += total_voting_power
        elif choice == "abstain":
            proposal.abstain_votes += total_voting_power

        proposal.total_votes += total_voting_power

        self.voting_records.append(vote_record)
        await self._save_vote_record(vote_record)
        await self._save_proposal(proposal)

        # Update member stats
        member.votes_cast += 1
        member.last_active = current_time
        await self._save_member(member)

        # Handle delegated votes
        if use_delegation and member.delegated_power > 0:
            for delegator_id in member.delegators:
                delegator_vote = VotingRecord(
                    vote_id=str(uuid.uuid4()),
                    proposal_id=proposal_id,
                    member_id=delegator_id,
                    choice=choice,
                    voting_power=self.members[delegator_id].voting_power,
                    timestamp=current_time,
                    delegated=True,
                    delegate_id=member_id,
                )
                self.voting_records.append(delegator_vote)
                await self._save_vote_record(delegator_vote)

        await self._log_governance_event(
            "vote_cast",
            actor_id=member_id,
            target_id=proposal_id,
            details={
                "choice": choice,
                "voting_power": total_voting_power,
                "delegated_votes": member.delegated_power if use_delegation else 0,
            },
        )

        logger.info(f"Vote cast: {member_id} voted {choice} on {proposal_id} with {total_voting_power} power")
        return vote_id

    # Background Tasks

    async def _proposal_lifecycle_manager(self):
        """Background task to manage proposal lifecycle transitions."""
        while self._running:
            try:
                current_time = datetime.utcnow()

                for proposal in self.proposals.values():
                    # Check for automatic phase transitions
                    if (
                        proposal.phase == ProposalPhase.REVIEW
                        and proposal.review_deadline
                        and current_time > proposal.review_deadline
                    ):
                        # Auto-advance if no reviewer response
                        if not proposal.council_approved:
                            proposal.council_approved = True
                        await self.advance_proposal_phase(proposal.proposal_id)

                    elif (
                        proposal.phase == ProposalPhase.DISCUSSION
                        and proposal.discussion_deadline
                        and current_time > proposal.discussion_deadline
                    ):
                        await self.advance_proposal_phase(proposal.proposal_id)

                    elif (
                        proposal.phase == ProposalPhase.VOTING
                        and proposal.voting_end
                        and current_time > proposal.voting_end
                    ):
                        await self.advance_proposal_phase(proposal.proposal_id)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in proposal lifecycle manager: {e}")
                await asyncio.sleep(60)

    async def _voting_monitor(self):
        """Monitor voting activity and detect issues."""
        while self._running:
            try:
                # Monitor active votes
                for proposal in self.proposals.values():
                    if proposal.phase == ProposalPhase.VOTING:
                        # Check voting participation
                        total_members = len([m for m in self.members.values() if m.can_vote])
                        participation_rate = proposal.participant_count / total_members if total_members > 0 else 0

                        # Alert on low participation
                        if participation_rate < 0.1:  # Less than 10% participation
                            logger.warning(
                                f"Low participation in proposal {proposal.proposal_id}: {participation_rate:.1%}"
                            )

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error(f"Error in voting monitor: {e}")
                await asyncio.sleep(300)

    async def _delegation_manager(self):
        """Manage vote delegations and expirations."""
        while self._running:
            try:
                current_time = datetime.utcnow()

                # Check for expired delegations
                for delegation in list(self.delegations.values()):
                    if delegation.active and delegation.expires_at and current_time > delegation.expires_at:
                        await self.revoke_delegation(delegation.delegator_id)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in delegation manager: {e}")
                await asyncio.sleep(300)

    async def _compliance_monitor(self):
        """Monitor governance for compliance violations."""
        while self._running:
            try:
                # Periodic compliance checks
                for proposal in self.proposals.values():
                    if proposal.phase in [ProposalPhase.VOTING, ProposalPhase.EXECUTION]:
                        # Check for unusual voting patterns
                        if proposal.total_votes > 0:
                            yes_rate = proposal.yes_votes / proposal.total_votes
                            if yes_rate > 0.95 or yes_rate < 0.05:
                                logger.warning(
                                    f"Unusual voting pattern in proposal {proposal.proposal_id}: {yes_rate:.1%} yes rate"
                                )

                await asyncio.sleep(7200)  # Check every 2 hours

            except Exception as e:
                logger.error(f"Error in compliance monitor: {e}")
                await asyncio.sleep(300)

    async def _governance_reporter(self):
        """Generate governance activity reports."""
        while self._running:
            try:
                # Generate weekly report
                report = await self.generate_governance_report()

                # Save report
                report_file = self.data_dir / f"governance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)

                logger.info(f"Generated governance report: {report_file}")

                await asyncio.sleep(604800)  # Weekly reporting

            except Exception as e:
                logger.error(f"Error in governance reporter: {e}")
                await asyncio.sleep(3600)

    async def _reputation_updater(self):
        """Update member reputation scores."""
        while self._running:
            try:
                for member in self.members.values():
                    # Calculate participation rate
                    total_proposals = len(
                        [
                            p
                            for p in self.proposals.values()
                            if p.phase in [ProposalPhase.COMPLETED, ProposalPhase.REJECTED]
                        ]
                    )
                    if total_proposals > 0:
                        member.participation_rate = member.votes_cast / total_proposals

                    # Update reputation based on participation
                    if member.participation_rate > 0.8:
                        member.reputation_score = min(200, member.reputation_score + 1)
                    elif member.participation_rate < 0.2:
                        member.reputation_score = max(50, member.reputation_score - 1)

                    await self._save_member(member)

                await asyncio.sleep(86400)  # Daily updates

            except Exception as e:
                logger.error(f"Error in reputation updater: {e}")
                await asyncio.sleep(3600)

    # Reporting and Analytics

    async def generate_governance_report(self) -> dict[str, Any]:
        """Generate comprehensive governance activity report."""

        # Calculate time periods
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        now - timedelta(days=30)

        # Member statistics
        total_members = len(self.members)
        active_members = len([m for m in self.members.values() if m.last_active > week_ago])
        verified_members = len([m for m in self.members.values() if m.verified])

        # Proposal statistics
        total_proposals = len(self.proposals)
        active_proposals = len([p for p in self.proposals.values() if p.phase == ProposalPhase.VOTING])
        recent_proposals = len([p for p in self.proposals.values() if p.created_at > week_ago])
        completed_proposals = len([p for p in self.proposals.values() if p.phase == ProposalPhase.COMPLETED])

        # Voting statistics
        recent_votes = len([v for v in self.voting_records if v.timestamp > week_ago])
        total_voting_power = sum(m.voting_power + m.delegated_power for m in self.members.values())
        delegated_power = sum(m.delegated_power for m in self.members.values())

        # Participation analysis
        avg_participation = 0.0
        if self.members:
            avg_participation = sum(m.participation_rate for m in self.members.values()) / len(self.members)

        return {
            "report_timestamp": now.isoformat(),
            "reporting_period": {"start": week_ago.isoformat(), "end": now.isoformat()},
            "membership": {
                "total_members": total_members,
                "active_members_week": active_members,
                "verified_members": verified_members,
                "verification_rate": verified_members / total_members if total_members > 0 else 0,
                "by_role": {
                    role.value: len([m for m in self.members.values() if m.role == role]) for role in MemberRole
                },
            },
            "proposals": {
                "total_proposals": total_proposals,
                "active_proposals": active_proposals,
                "recent_proposals_week": recent_proposals,
                "completed_proposals": completed_proposals,
                "completion_rate": completed_proposals / total_proposals if total_proposals > 0 else 0,
                "by_category": {
                    cat.value: len([p for p in self.proposals.values() if p.category == cat])
                    for cat in ProposalCategory
                },
                "by_phase": {
                    phase.value: len([p for p in self.proposals.values() if p.phase == phase])
                    for phase in ProposalPhase
                },
            },
            "voting": {
                "total_votes_cast": len(self.voting_records),
                "recent_votes_week": recent_votes,
                "total_voting_power": total_voting_power,
                "delegated_power": delegated_power,
                "delegation_rate": delegated_power / total_voting_power if total_voting_power > 0 else 0,
                "avg_participation_rate": avg_participation,
            },
            "delegations": {
                "active_delegations": len([d for d in self.delegations.values() if d.active]),
                "total_delegations": len(self.delegations),
            },
            "compliance": {
                "proposals_with_compliance_check": len(
                    [p for p in self.proposals.values() if self.config["compliance"]["proposal_compliance_check"]]
                ),
                "audit_trail_entries": await self._count_audit_entries(),
            },
        }

    # Database Operations

    async def _save_member(self, member: DAOMember):
        """Save member to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO members
            (member_id, wallet_address, role, voting_power, delegated_power, delegates_to,
             proposals_created, votes_cast, participation_rate, reputation_score,
             joined_at, last_active, verified, can_create_proposals, can_vote, can_delegate, delegators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                member.member_id,
                member.wallet_address,
                member.role.value,
                member.voting_power,
                member.delegated_power,
                member.delegates_to,
                member.proposals_created,
                member.votes_cast,
                member.participation_rate,
                member.reputation_score,
                member.joined_at.isoformat(),
                member.last_active.isoformat(),
                member.verified,
                member.can_create_proposals,
                member.can_vote,
                member.can_delegate,
                json.dumps(list(member.delegators)),
            ),
        )

        conn.commit()
        conn.close()

    async def _save_proposal(self, proposal: GovernanceProposal):
        """Save proposal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO proposals
            (proposal_id, title, description, category, proposer_id, summary, detailed_specification,
             implementation_plan, success_criteria, risk_assessment, phase, created_at, review_deadline,
             discussion_deadline, voting_start, voting_end, quorum_required, approval_threshold, voting_type,
             reviewer_assigned, review_comments, council_approved, total_votes, yes_votes, no_votes,
             abstain_votes, participant_count, execution_status, execution_results, tags, attachments, dependencies)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                proposal.proposal_id,
                proposal.title,
                proposal.description,
                proposal.category.value,
                proposal.proposer_id,
                proposal.summary,
                proposal.detailed_specification,
                proposal.implementation_plan,
                proposal.success_criteria,
                proposal.risk_assessment,
                proposal.phase.value,
                proposal.created_at.isoformat(),
                proposal.review_deadline.isoformat() if proposal.review_deadline else None,
                proposal.discussion_deadline.isoformat() if proposal.discussion_deadline else None,
                proposal.voting_start.isoformat() if proposal.voting_start else None,
                proposal.voting_end.isoformat() if proposal.voting_end else None,
                proposal.quorum_required,
                proposal.approval_threshold,
                proposal.voting_type,
                proposal.reviewer_assigned,
                json.dumps(proposal.review_comments),
                proposal.council_approved,
                proposal.total_votes,
                proposal.yes_votes,
                proposal.no_votes,
                proposal.abstain_votes,
                proposal.participant_count,
                proposal.execution_status,
                json.dumps(proposal.execution_results),
                json.dumps(proposal.tags),
                json.dumps(proposal.attachments),
                json.dumps(proposal.dependencies),
            ),
        )

        conn.commit()
        conn.close()

    async def _save_vote_record(self, vote: VotingRecord):
        """Save vote record to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO voting_records
            (vote_id, proposal_id, member_id, choice, voting_power, timestamp, delegated, delegate_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                vote.vote_id,
                vote.proposal_id,
                vote.member_id,
                vote.choice,
                vote.voting_power,
                vote.timestamp.isoformat(),
                vote.delegated,
                vote.delegate_id,
            ),
        )

        conn.commit()
        conn.close()

    async def _save_delegation(self, delegation: DelegationRecord):
        """Save delegation to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO delegations
            (delegation_id, delegator_id, delegate_id, voting_power, created_at, expires_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                delegation.delegation_id,
                delegation.delegator_id,
                delegation.delegate_id,
                delegation.voting_power,
                delegation.created_at.isoformat(),
                delegation.expires_at.isoformat() if delegation.expires_at else None,
                delegation.active,
            ),
        )

        conn.commit()
        conn.close()

    async def _log_governance_event(
        self, event_type: str, actor_id: str | None = None, target_id: str | None = None, details: dict[str, Any] = None
    ):
        """Log governance event for audit trail."""
        audit_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO governance_audit_log
            (audit_id, event_type, timestamp, actor_id, target_id, details, compliance_checked)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
        """,
            (audit_id, event_type, actor_id, target_id, json.dumps(details) if details else None, True),
        )

        conn.commit()
        conn.close()

    async def _count_audit_entries(self) -> int:
        """Count audit log entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM governance_audit_log")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    # Public Query Methods

    async def get_governance_status(self) -> dict[str, Any]:
        """Get current governance system status."""
        return await self.generate_governance_report()

    async def get_member_info(self, member_id: str) -> dict[str, Any] | None:
        """Get member information."""
        if member_id not in self.members:
            return None

        member = self.members[member_id]
        return {
            "member_id": member.member_id,
            "wallet_address": member.wallet_address,
            "role": member.role.value,
            "voting_power": member.voting_power,
            "delegated_power": member.delegated_power,
            "delegates_to": member.delegates_to,
            "proposals_created": member.proposals_created,
            "votes_cast": member.votes_cast,
            "participation_rate": member.participation_rate,
            "reputation_score": member.reputation_score,
            "verified": member.verified,
            "joined_at": member.joined_at.isoformat(),
            "last_active": member.last_active.isoformat(),
        }

    async def get_proposal_details(self, proposal_id: str) -> dict[str, Any] | None:
        """Get detailed proposal information."""
        if proposal_id not in self.proposals:
            return None

        proposal = self.proposals[proposal_id]
        votes = [v for v in self.voting_records if v.proposal_id == proposal_id]

        return {
            "proposal_id": proposal.proposal_id,
            "title": proposal.title,
            "description": proposal.description,
            "category": proposal.category.value,
            "proposer_id": proposal.proposer_id,
            "phase": proposal.phase.value,
            "created_at": proposal.created_at.isoformat(),
            "voting_start": proposal.voting_start.isoformat() if proposal.voting_start else None,
            "voting_end": proposal.voting_end.isoformat() if proposal.voting_end else None,
            "quorum_required": proposal.quorum_required,
            "approval_threshold": proposal.approval_threshold,
            "total_votes": proposal.total_votes,
            "yes_votes": proposal.yes_votes,
            "no_votes": proposal.no_votes,
            "abstain_votes": proposal.abstain_votes,
            "participant_count": proposal.participant_count,
            "execution_status": proposal.execution_status,
            "votes": [
                {
                    "member_id": v.member_id,
                    "choice": v.choice,
                    "voting_power": v.voting_power,
                    "timestamp": v.timestamp.isoformat(),
                    "delegated": v.delegated,
                }
                for v in votes
            ],
        }

    async def get_active_proposals(self) -> list[dict[str, Any]]:
        """Get all active proposals."""
        active_proposals = [
            p
            for p in self.proposals.values()
            if p.phase in [ProposalPhase.REVIEW, ProposalPhase.DISCUSSION, ProposalPhase.VOTING]
        ]

        return [
            {
                "proposal_id": p.proposal_id,
                "title": p.title,
                "category": p.category.value,
                "phase": p.phase.value,
                "created_at": p.created_at.isoformat(),
                "voting_end": p.voting_end.isoformat() if p.voting_end else None,
                "total_votes": p.total_votes,
                "quorum_required": p.quorum_required,
            }
            for p in active_proposals
        ]
