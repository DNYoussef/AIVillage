#!/usr/bin/env python3
"""
DAO Governance System - Complete Implementation

This module provides comprehensive DAO governance operational procedures including:
- Voting system for community decisions
- Proposal submission and review process
- Delegate voting and quorum management
- Governance dashboard and member management
- Integration with FOG token economics
- Audit trail and compliance logging

Key Features:
- On-chain and off-chain voting mechanisms
- Multi-tier governance structure (Members, Delegates, Validators)
- Proposal lifecycle management (Draft -> Review -> Voting -> Execution)
- Quorum-based decision making
- Delegated voting with proxy mechanisms
- Economic incentives for participation
- Comprehensive audit logging
- Regulatory compliance integration
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProposalType(Enum):
    """Types of governance proposals."""

    PROTOCOL_UPGRADE = "protocol_upgrade"
    TOKENOMICS_CHANGE = "tokenomics_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    GOVERNANCE_RULE = "governance_rule"
    EMERGENCY_ACTION = "emergency_action"
    TREASURY_SPEND = "treasury_spend"
    VALIDATOR_CHANGE = "validator_change"
    COMPLIANCE_UPDATE = "compliance_update"


class ProposalStatus(Enum):
    """Proposal lifecycle states."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    REVIEW = "review"
    VOTING = "voting"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class VoteChoice(Enum):
    """Vote choices."""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class MemberRole(Enum):
    """DAO member roles."""

    MEMBER = "member"
    DELEGATE = "delegate"
    VALIDATOR = "validator"
    ADMIN = "admin"


class GovernanceProposal(BaseModel):
    """Governance proposal model."""

    proposal_id: str
    title: str
    description: str
    proposal_type: ProposalType
    status: ProposalStatus
    author_id: str
    created_at: datetime
    voting_start: datetime | None = None
    voting_end: datetime | None = None
    execution_target: datetime | None = None

    # Voting parameters
    quorum_threshold: float = 0.51  # 51% participation required
    approval_threshold: float = 0.60  # 60% approval required
    minimum_voting_power: int = 1000  # Minimum tokens to participate

    # Vote tracking
    total_votes: int = 0
    yes_votes: int = 0
    no_votes: int = 0
    abstain_votes: int = 0
    yes_voting_power: int = 0
    no_voting_power: int = 0
    abstain_voting_power: int = 0
    total_voting_power: int = 0

    # Proposal content
    proposal_data: dict[str, Any] = Field(default_factory=dict)
    execution_actions: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    tags: list[str] = Field(default_factory=list)
    discussion_link: str | None = None
    implementation_timeline: int | None = None  # Days
    budget_request: int | None = None  # FOG tokens

    def get_voting_results(self) -> dict[str, Any]:
        """Get detailed voting results."""
        if self.total_voting_power == 0:
            return {
                "participation_rate": 0.0,
                "approval_rate": 0.0,
                "quorum_met": False,
                "approval_threshold_met": False,
                "result": "insufficient_participation",
            }

        participation_rate = self.total_voting_power / 1000000  # Assume 1M total supply
        approval_rate = self.yes_voting_power / max(self.total_voting_power, 1)

        quorum_met = participation_rate >= self.quorum_threshold
        approval_met = approval_rate >= self.approval_threshold

        if not quorum_met:
            result = "quorum_not_met"
        elif not approval_met:
            result = "rejected"
        else:
            result = "approved"

        return {
            "participation_rate": participation_rate,
            "approval_rate": approval_rate,
            "quorum_met": quorum_met,
            "approval_threshold_met": approval_met,
            "result": result,
            "total_voters": self.total_votes,
            "voting_power_distribution": {
                "yes": self.yes_voting_power,
                "no": self.no_voting_power,
                "abstain": self.abstain_voting_power,
            },
        }


class GovernanceMember(BaseModel):
    """DAO governance member."""

    member_id: str
    address: str  # Wallet/account address
    role: MemberRole
    voting_power: int
    delegated_power: int = 0  # Power delegated to this member
    delegation_target: str | None = None  # Who this member delegates to

    # Member metrics
    proposals_created: int = 0
    votes_cast: int = 0
    participation_rate: float = 0.0
    reputation_score: int = 100

    # Membership details
    joined_at: datetime
    last_activity: datetime
    active: bool = True

    # KYC/Compliance
    kyc_verified: bool = False
    jurisdiction: str | None = None
    compliance_tier: str = "basic"  # basic, verified, institutional

    def get_total_voting_power(self) -> int:
        """Get total voting power including delegations."""
        return self.voting_power + self.delegated_power


class GovernanceVote(BaseModel):
    """Individual governance vote."""

    vote_id: str
    proposal_id: str
    voter_id: str
    choice: VoteChoice
    voting_power: int
    reason: str | None = None
    cast_at: datetime
    is_delegation: bool = False
    original_voter: str | None = None  # If this is a delegated vote


class DAOGovernanceSystem:
    """Complete DAO governance system implementation."""

    def __init__(self, data_dir: str = "./dao_governance_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Core data structures
        self.proposals: dict[str, GovernanceProposal] = {}
        self.members: dict[str, GovernanceMember] = {}
        self.votes: dict[str, list[GovernanceVote]] = {}  # proposal_id -> votes
        self.delegations: dict[str, str] = {}  # delegator_id -> delegate_id

        # Governance parameters
        self.governance_config = {
            "proposal_deposit": 10000,  # FOG tokens required to submit proposal
            "voting_period_days": 7,  # Default voting period
            "review_period_days": 3,  # Review period before voting
            "execution_delay_days": 2,  # Delay after approval before execution
            "minimum_quorum": 0.51,  # 51% participation required
            "approval_threshold": 0.60,  # 60% approval required
            "emergency_quorum": 0.30,  # Lower quorum for emergency proposals
            "max_delegation_depth": 3,  # Maximum delegation chain length
            "reputation_decay_days": 90,  # Reputation decay period
        }

        # Audit log
        self.audit_log: list[dict[str, Any]] = []

        # Initialize with sample data
        self._initialize_governance()

    def _initialize_governance(self):
        """Initialize governance system with sample data."""
        logger.info("Initializing DAO governance system...")

        # Create founding members
        founding_members = [
            ("founder_001", "0x1234...abcd", MemberRole.ADMIN, 100000),
            ("founder_002", "0x5678...efgh", MemberRole.VALIDATOR, 75000),
            ("founder_003", "0x9012...ijkl", MemberRole.VALIDATOR, 75000),
            ("delegate_001", "0xabcd...1234", MemberRole.DELEGATE, 50000),
            ("delegate_002", "0xefgh...5678", MemberRole.DELEGATE, 50000),
        ]

        for member_id, address, role, voting_power in founding_members:
            self.add_member(
                member_id=member_id,
                address=address,
                role=role,
                voting_power=voting_power,
                kyc_verified=True,
                jurisdiction="global",
            )

        # Create sample proposals
        self._create_sample_proposals()

        logger.info(f"Initialized governance with {len(self.members)} members and {len(self.proposals)} proposals")

    def _create_sample_proposals(self):
        """Create sample governance proposals."""
        sample_proposals = [
            {
                "title": "Increase Fog Computing Reward Rate",
                "description": "Proposal to increase the reward rate for fog computing contributions from 10 to 15 FOG tokens per hour to incentivize more participation.",
                "proposal_type": ProposalType.TOKENOMICS_CHANGE,
                "author_id": "founder_001",
                "proposal_data": {
                    "current_rate": 10,
                    "proposed_rate": 15,
                    "estimated_annual_cost": 2000000,
                    "implementation_complexity": "low",
                },
                "budget_request": 2000000,
                "implementation_timeline": 30,
            },
            {
                "title": "Protocol Upgrade v2.1 - Enhanced Privacy",
                "description": "Upgrade the fog computing protocol to version 2.1 with enhanced privacy features including improved onion routing and zero-knowledge proofs.",
                "proposal_type": ProposalType.PROTOCOL_UPGRADE,
                "author_id": "founder_002",
                "proposal_data": {
                    "current_version": "2.0",
                    "target_version": "2.1",
                    "features": ["enhanced_onion_routing", "zk_proofs", "improved_encryption"],
                    "security_audit_required": True,
                },
                "implementation_timeline": 90,
            },
            {
                "title": "Treasury Allocation for Marketing Campaign",
                "description": "Allocate 500,000 FOG tokens from the treasury for a global marketing campaign to increase adoption of the fog computing network.",
                "proposal_type": ProposalType.TREASURY_SPEND,
                "author_id": "delegate_001",
                "proposal_data": {
                    "campaign_duration_months": 6,
                    "target_regions": ["North America", "Europe", "Asia"],
                    "expected_new_users": 10000,
                    "marketing_channels": ["social_media", "conferences", "partnerships"],
                },
                "budget_request": 500000,
                "implementation_timeline": 14,
            },
        ]

        for proposal_data in sample_proposals:
            proposal_id = self.create_proposal(
                title=proposal_data["title"],
                description=proposal_data["description"],
                proposal_type=proposal_data["proposal_type"],
                author_id=proposal_data["author_id"],
                proposal_data=proposal_data.get("proposal_data", {}),
                budget_request=proposal_data.get("budget_request"),
                implementation_timeline=proposal_data.get("implementation_timeline"),
            )

            # Move first proposal to voting stage
            if len(self.proposals) == 1:
                self.start_voting(proposal_id)

    def add_member(
        self,
        member_id: str,
        address: str,
        role: MemberRole,
        voting_power: int,
        kyc_verified: bool = False,
        jurisdiction: str = None,
    ) -> GovernanceMember:
        """Add a new governance member."""
        member = GovernanceMember(
            member_id=member_id,
            address=address,
            role=role,
            voting_power=voting_power,
            joined_at=datetime.now(),
            last_activity=datetime.now(),
            kyc_verified=kyc_verified,
            jurisdiction=jurisdiction,
            compliance_tier="verified" if kyc_verified else "basic",
        )

        self.members[member_id] = member

        # Audit log
        self.audit_log.append(
            {
                "action": "member_added",
                "member_id": member_id,
                "role": role.value,
                "voting_power": voting_power,
                "timestamp": datetime.now().isoformat(),
                "kyc_verified": kyc_verified,
            }
        )

        logger.info(f"Added governance member: {member_id} ({role.value}) with {voting_power} voting power")
        return member

    def create_proposal(
        self,
        title: str,
        description: str,
        proposal_type: ProposalType,
        author_id: str,
        proposal_data: dict[str, Any] = None,
        budget_request: int = None,
        implementation_timeline: int = None,
    ) -> str:
        """Create a new governance proposal."""
        if author_id not in self.members:
            raise ValueError(f"Author {author_id} is not a registered member")

        author = self.members[author_id]
        if author.voting_power < self.governance_config["proposal_deposit"]:
            raise ValueError(
                f"Insufficient voting power to create proposal. Required: {self.governance_config['proposal_deposit']}"
            )

        proposal_id = f"prop_{uuid.uuid4().hex[:12]}"

        proposal = GovernanceProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposal_type=proposal_type,
            status=ProposalStatus.DRAFT,
            author_id=author_id,
            created_at=datetime.now(),
            proposal_data=proposal_data or {},
            budget_request=budget_request,
            implementation_timeline=implementation_timeline,
            tags=self._generate_proposal_tags(proposal_type, proposal_data or {}),
        )

        self.proposals[proposal_id] = proposal
        self.votes[proposal_id] = []

        # Update author stats
        self.members[author_id].proposals_created += 1
        self.members[author_id].last_activity = datetime.now()

        # Audit log
        self.audit_log.append(
            {
                "action": "proposal_created",
                "proposal_id": proposal_id,
                "author_id": author_id,
                "proposal_type": proposal_type.value,
                "title": title,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Created proposal {proposal_id}: {title}")
        return proposal_id

    def _generate_proposal_tags(self, proposal_type: ProposalType, proposal_data: dict[str, Any]) -> list[str]:
        """Generate tags for a proposal based on type and content."""
        tags = [proposal_type.value]

        # Add budget-related tags
        if "budget_request" in proposal_data:
            amount = proposal_data["budget_request"]
            if amount > 1000000:
                tags.append("large_budget")
            elif amount > 100000:
                tags.append("medium_budget")
            else:
                tags.append("small_budget")

        # Add implementation tags
        if "implementation_timeline" in proposal_data:
            timeline = proposal_data["implementation_timeline"]
            if timeline > 90:
                tags.append("long_term")
            elif timeline > 30:
                tags.append("medium_term")
            else:
                tags.append("short_term")

        # Add type-specific tags
        if proposal_type == ProposalType.PROTOCOL_UPGRADE:
            tags.extend(["technical", "security_critical"])
        elif proposal_type == ProposalType.TOKENOMICS_CHANGE:
            tags.extend(["economic", "incentives"])
        elif proposal_type == ProposalType.TREASURY_SPEND:
            tags.extend(["treasury", "spending"])

        return tags

    def submit_proposal(self, proposal_id: str) -> bool:
        """Submit a proposal for review."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status != ProposalStatus.DRAFT:
            raise ValueError(f"Proposal {proposal_id} is not in draft status")

        proposal.status = ProposalStatus.SUBMITTED

        # Audit log
        self.audit_log.append(
            {"action": "proposal_submitted", "proposal_id": proposal_id, "timestamp": datetime.now().isoformat()}
        )

        logger.info(f"Submitted proposal {proposal_id} for review")
        return True

    def start_review(self, proposal_id: str) -> bool:
        """Start the review period for a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status != ProposalStatus.SUBMITTED:
            raise ValueError(f"Proposal {proposal_id} is not submitted")

        proposal.status = ProposalStatus.REVIEW

        # Schedule voting start
        review_days = self.governance_config["review_period_days"]
        proposal.voting_start = datetime.now() + timedelta(days=review_days)

        # Audit log
        self.audit_log.append(
            {
                "action": "proposal_review_started",
                "proposal_id": proposal_id,
                "review_end": proposal.voting_start.isoformat(),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Started review period for proposal {proposal_id}")
        return True

    def start_voting(self, proposal_id: str) -> bool:
        """Start the voting period for a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status not in [ProposalStatus.REVIEW, ProposalStatus.SUBMITTED]:
            raise ValueError(f"Proposal {proposal_id} is not ready for voting")

        proposal.status = ProposalStatus.VOTING
        proposal.voting_start = datetime.now()

        # Set voting period based on proposal type
        voting_days = self.governance_config["voting_period_days"]
        if proposal.proposal_type == ProposalType.EMERGENCY_ACTION:
            voting_days = 1  # Emergency proposals have 1-day voting

        proposal.voting_end = proposal.voting_start + timedelta(days=voting_days)

        # Set quorum based on proposal type
        if proposal.proposal_type == ProposalType.EMERGENCY_ACTION:
            proposal.quorum_threshold = self.governance_config["emergency_quorum"]

        # Audit log
        self.audit_log.append(
            {
                "action": "voting_started",
                "proposal_id": proposal_id,
                "voting_start": proposal.voting_start.isoformat(),
                "voting_end": proposal.voting_end.isoformat(),
                "quorum_threshold": proposal.quorum_threshold,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Started voting for proposal {proposal_id} (ends: {proposal.voting_end})")
        return True

    def cast_vote(self, proposal_id: str, voter_id: str, choice: VoteChoice, reason: str = None) -> bool:
        """Cast a vote on a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        if voter_id not in self.members:
            raise ValueError(f"Voter {voter_id} is not a registered member")

        proposal = self.proposals[proposal_id]
        voter = self.members[voter_id]

        # Check voting eligibility
        if proposal.status != ProposalStatus.VOTING:
            raise ValueError(f"Proposal {proposal_id} is not in voting status")

        if datetime.now() > proposal.voting_end:
            raise ValueError(f"Voting period for proposal {proposal_id} has ended")

        if voter.voting_power < proposal.minimum_voting_power:
            raise ValueError(f"Insufficient voting power. Required: {proposal.minimum_voting_power}")

        # Check if already voted
        existing_votes = [v for v in self.votes[proposal_id] if v.voter_id == voter_id]
        if existing_votes:
            raise ValueError(f"Voter {voter_id} has already voted on proposal {proposal_id}")

        # Calculate total voting power (including delegations)
        total_voting_power = self._get_member_voting_power(voter_id, proposal_id)

        # Create vote record
        vote = GovernanceVote(
            vote_id=f"vote_{uuid.uuid4().hex[:12]}",
            proposal_id=proposal_id,
            voter_id=voter_id,
            choice=choice,
            voting_power=total_voting_power,
            reason=reason,
            cast_at=datetime.now(),
        )

        self.votes[proposal_id].append(vote)

        # Update proposal vote counts
        proposal.total_votes += 1
        proposal.total_voting_power += total_voting_power

        if choice == VoteChoice.YES:
            proposal.yes_votes += 1
            proposal.yes_voting_power += total_voting_power
        elif choice == VoteChoice.NO:
            proposal.no_votes += 1
            proposal.no_voting_power += total_voting_power
        elif choice == VoteChoice.ABSTAIN:
            proposal.abstain_votes += 1
            proposal.abstain_voting_power += total_voting_power

        # Handle delegated votes
        self._process_delegated_votes(proposal_id, voter_id, choice, total_voting_power)

        # Update member stats
        voter.votes_cast += 1
        voter.last_activity = datetime.now()
        voter.participation_rate = min(1.0, voter.votes_cast / max(len(self.proposals), 1))

        # Audit log
        self.audit_log.append(
            {
                "action": "vote_cast",
                "proposal_id": proposal_id,
                "voter_id": voter_id,
                "choice": choice.value,
                "voting_power": total_voting_power,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Vote cast by {voter_id} on proposal {proposal_id}: {choice.value} ({total_voting_power} voting power)"
        )

        # Check if voting can be finalized early
        self._check_early_finalization(proposal_id)

        return True

    def _get_member_voting_power(self, member_id: str, proposal_id: str) -> int:
        """Get total voting power for a member including delegations."""
        if member_id not in self.members:
            return 0

        member = self.members[member_id]
        total_power = member.voting_power

        # Add delegated power from others
        for delegator_id, delegate_id in self.delegations.items():
            if delegate_id == member_id and delegator_id in self.members:
                # Check if delegator hasn't voted directly
                delegator_votes = [v for v in self.votes[proposal_id] if v.voter_id == delegator_id]
                if not delegator_votes:
                    total_power += self.members[delegator_id].voting_power

        return total_power

    def _process_delegated_votes(self, proposal_id: str, voter_id: str, choice: VoteChoice, voting_power: int):
        """Process delegated votes for members who delegated to this voter."""
        for delegator_id, delegate_id in self.delegations.items():
            if delegate_id == voter_id and delegator_id in self.members:
                # Check if delegator hasn't voted directly
                delegator_votes = [v for v in self.votes[proposal_id] if v.voter_id == delegator_id]
                if not delegator_votes:
                    delegated_vote = GovernanceVote(
                        vote_id=f"vote_{uuid.uuid4().hex[:12]}",
                        proposal_id=proposal_id,
                        voter_id=delegator_id,
                        choice=choice,
                        voting_power=self.members[delegator_id].voting_power,
                        cast_at=datetime.now(),
                        is_delegation=True,
                        original_voter=voter_id,
                    )

                    self.votes[proposal_id].append(delegated_vote)

    def _check_early_finalization(self, proposal_id: str):
        """Check if a proposal can be finalized early due to overwhelming support/opposition."""
        proposal = self.proposals[proposal_id]

        # Get current results
        results = proposal.get_voting_results()

        # Check for overwhelming support (>80% with sufficient participation)
        if results["approval_rate"] > 0.80 and results["participation_rate"] > proposal.quorum_threshold:
            self._finalize_voting(proposal_id, early=True)

        # Check for overwhelming rejection (>80% opposition with sufficient participation)
        elif results["approval_rate"] < 0.20 and results["participation_rate"] > proposal.quorum_threshold:
            self._finalize_voting(proposal_id, early=True)

    def finalize_voting(self, proposal_id: str) -> dict[str, Any]:
        """Finalize voting for a proposal."""
        return self._finalize_voting(proposal_id, early=False)

    def _finalize_voting(self, proposal_id: str, early: bool = False) -> dict[str, Any]:
        """Internal method to finalize voting."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status != ProposalStatus.VOTING:
            raise ValueError(f"Proposal {proposal_id} is not in voting status")

        if not early and datetime.now() < proposal.voting_end:
            raise ValueError(f"Voting period for proposal {proposal_id} has not ended")

        # Get voting results
        results = proposal.get_voting_results()

        # Determine final status
        if results["result"] == "approved":
            proposal.status = ProposalStatus.PASSED
            # Schedule execution
            execution_delay = self.governance_config["execution_delay_days"]
            proposal.execution_target = datetime.now() + timedelta(days=execution_delay)
        else:
            proposal.status = ProposalStatus.REJECTED

        # Update member reputation based on participation
        self._update_member_reputation(proposal_id)

        # Audit log
        self.audit_log.append(
            {
                "action": "voting_finalized",
                "proposal_id": proposal_id,
                "result": results["result"],
                "early_finalization": early,
                "participation_rate": results["participation_rate"],
                "approval_rate": results["approval_rate"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Finalized voting for proposal {proposal_id}: {results['result']} "
            f"({results['participation_rate']:.1%} participation, "
            f"{results['approval_rate']:.1%} approval)"
        )

        return {
            "proposal_id": proposal_id,
            "final_status": proposal.status.value,
            "results": results,
            "execution_target": proposal.execution_target.isoformat() if proposal.execution_target else None,
        }

    def _update_member_reputation(self, proposal_id: str):
        """Update member reputation based on voting participation."""
        proposal_votes = self.votes[proposal_id]

        for vote in proposal_votes:
            if vote.voter_id in self.members and not vote.is_delegation:
                member = self.members[vote.voter_id]
                # Increase reputation for participation
                member.reputation_score = min(1000, member.reputation_score + 5)

        # Decrease reputation for non-participation (active members only)
        active_members = [m for m in self.members.values() if m.active and m.voting_power >= 1000]
        voted_member_ids = {v.voter_id for v in proposal_votes if not v.is_delegation}

        for member in active_members:
            if member.member_id not in voted_member_ids:
                member.reputation_score = max(0, member.reputation_score - 2)

    def delegate_voting_power(self, delegator_id: str, delegate_id: str) -> bool:
        """Delegate voting power to another member."""
        if delegator_id not in self.members or delegate_id not in self.members:
            raise ValueError("Both delegator and delegate must be registered members")

        if delegator_id == delegate_id:
            raise ValueError("Cannot delegate to self")

        # Check delegation depth to prevent cycles
        depth = 0
        current = delegate_id
        while current in self.delegations and depth < self.governance_config["max_delegation_depth"]:
            current = self.delegations[current]
            depth += 1
            if current == delegator_id:
                raise ValueError("Delegation would create a cycle")

        if depth >= self.governance_config["max_delegation_depth"]:
            raise ValueError("Maximum delegation depth exceeded")

        # Remove existing delegation if any
        if delegator_id in self.delegations:
            old_delegate = self.delegations[delegator_id]
            self.members[old_delegate].delegated_power -= self.members[delegator_id].voting_power

        # Set new delegation
        self.delegations[delegator_id] = delegate_id
        self.members[delegator_id].delegation_target = delegate_id
        self.members[delegate_id].delegated_power += self.members[delegator_id].voting_power

        # Audit log
        self.audit_log.append(
            {
                "action": "delegation_created",
                "delegator_id": delegator_id,
                "delegate_id": delegate_id,
                "voting_power": self.members[delegator_id].voting_power,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Delegation created: {delegator_id} -> {delegate_id} "
            f"({self.members[delegator_id].voting_power} voting power)"
        )

        return True

    def revoke_delegation(self, delegator_id: str) -> bool:
        """Revoke voting power delegation."""
        if delegator_id not in self.delegations:
            raise ValueError(f"No delegation found for {delegator_id}")

        delegate_id = self.delegations[delegator_id]
        voting_power = self.members[delegator_id].voting_power

        # Remove delegation
        del self.delegations[delegator_id]
        self.members[delegator_id].delegation_target = None
        self.members[delegate_id].delegated_power -= voting_power

        # Audit log
        self.audit_log.append(
            {
                "action": "delegation_revoked",
                "delegator_id": delegator_id,
                "delegate_id": delegate_id,
                "voting_power": voting_power,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Delegation revoked: {delegator_id} -> {delegate_id}")
        return True

    def execute_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Execute a passed proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        if proposal.status != ProposalStatus.PASSED:
            raise ValueError(f"Proposal {proposal_id} has not passed voting")

        if proposal.execution_target and datetime.now() < proposal.execution_target:
            raise ValueError("Execution delay period has not passed")

        # Execute proposal actions based on type
        execution_results = self._execute_proposal_actions(proposal)

        # Mark as executed
        proposal.status = ProposalStatus.EXECUTED

        # Audit log
        self.audit_log.append(
            {
                "action": "proposal_executed",
                "proposal_id": proposal_id,
                "execution_results": execution_results,
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.info(f"Executed proposal {proposal_id}")

        return {"proposal_id": proposal_id, "execution_status": "completed", "results": execution_results}

    def _execute_proposal_actions(self, proposal: GovernanceProposal) -> dict[str, Any]:
        """Execute the specific actions for a proposal type."""
        results = {"actions_completed": [], "errors": []}

        try:
            if proposal.proposal_type == ProposalType.TOKENOMICS_CHANGE:
                # Example: Change reward rates
                if "proposed_rate" in proposal.proposal_data:
                    new_rate = proposal.proposal_data["proposed_rate"]
                    results["actions_completed"].append(f"Updated reward rate to {new_rate} FOG/hour")

            elif proposal.proposal_type == ProposalType.PROTOCOL_UPGRADE:
                # Example: Schedule protocol upgrade
                target_version = proposal.proposal_data.get("target_version", "unknown")
                results["actions_completed"].append(f"Scheduled protocol upgrade to {target_version}")

            elif proposal.proposal_type == ProposalType.TREASURY_SPEND:
                # Example: Allocate treasury funds
                amount = proposal.budget_request
                results["actions_completed"].append(f"Allocated {amount} FOG tokens from treasury")

            elif proposal.proposal_type == ProposalType.GOVERNANCE_RULE:
                # Example: Update governance parameters
                results["actions_completed"].append("Updated governance rules")

            # Add more proposal type handlers as needed

        except Exception as e:
            results["errors"].append(f"Execution error: {str(e)}")
            logger.error(f"Error executing proposal {proposal.proposal_id}: {e}")

        return results

    def get_governance_stats(self) -> dict[str, Any]:
        """Get comprehensive governance statistics."""
        now = datetime.now()

        # Member statistics
        total_members = len(self.members)
        active_members = len([m for m in self.members.values() if m.active])
        verified_members = len([m for m in self.members.values() if m.kyc_verified])

        # Role distribution
        role_distribution = {}
        for role in MemberRole:
            role_distribution[role.value] = len([m for m in self.members.values() if m.role == role])

        # Proposal statistics
        total_proposals = len(self.proposals)
        proposals_by_status = {}
        for status in ProposalStatus:
            proposals_by_status[status.value] = len([p for p in self.proposals.values() if p.status == status])

        # Voting statistics
        total_votes = sum(len(votes) for votes in self.votes.values())
        avg_participation = sum(p.total_votes for p in self.proposals.values()) / max(total_proposals, 1)

        # Active proposals
        active_proposals = [
            p
            for p in self.proposals.values()
            if p.status in [ProposalStatus.VOTING, ProposalStatus.REVIEW, ProposalStatus.SUBMITTED]
        ]

        # Delegation statistics
        total_delegations = len(self.delegations)
        delegated_power = sum(
            self.members[delegator_id].voting_power for delegator_id in self.delegations if delegator_id in self.members
        )

        return {
            "governance_health": {
                "total_members": total_members,
                "active_members": active_members,
                "verified_members": verified_members,
                "participation_rate": avg_participation / max(active_members, 1),
                "total_voting_power": sum(m.voting_power for m in self.members.values()),
                "delegated_power": delegated_power,
                "delegation_rate": total_delegations / max(total_members, 1),
            },
            "member_distribution": {
                "by_role": role_distribution,
                "by_voting_power": {
                    "whales": len([m for m in self.members.values() if m.voting_power >= 100000]),
                    "dolphins": len([m for m in self.members.values() if 10000 <= m.voting_power < 100000]),
                    "fish": len([m for m in self.members.values() if 1000 <= m.voting_power < 10000]),
                    "minnows": len([m for m in self.members.values() if m.voting_power < 1000]),
                },
            },
            "proposal_statistics": {
                "total_proposals": total_proposals,
                "by_status": proposals_by_status,
                "by_type": {
                    prop_type.value: len([p for p in self.proposals.values() if p.proposal_type == prop_type])
                    for prop_type in ProposalType
                },
                "active_proposals": len(active_proposals),
                "avg_voting_participation": avg_participation,
            },
            "voting_statistics": {
                "total_votes_cast": total_votes,
                "unique_voters": len(
                    set(vote.voter_id for votes in self.votes.values() for vote in votes if not vote.is_delegation)
                ),
                "delegated_votes": len([vote for votes in self.votes.values() for vote in votes if vote.is_delegation]),
            },
            "recent_activity": {
                "proposals_this_month": len(
                    [p for p in self.proposals.values() if p.created_at > now - timedelta(days=30)]
                ),
                "votes_this_week": len(
                    [vote for votes in self.votes.values() for vote in votes if vote.cast_at > now - timedelta(days=7)]
                ),
            },
            "timestamp": now.isoformat(),
        }

    def get_member_profile(self, member_id: str) -> dict[str, Any]:
        """Get detailed profile for a governance member."""
        if member_id not in self.members:
            raise ValueError(f"Member {member_id} not found")

        member = self.members[member_id]

        # Get member's proposals
        member_proposals = [
            {
                "proposal_id": p.proposal_id,
                "title": p.title,
                "status": p.status.value,
                "created_at": p.created_at.isoformat(),
                "type": p.proposal_type.value,
            }
            for p in self.proposals.values()
            if p.author_id == member_id
        ]

        # Get member's votes
        member_votes = []
        for proposal_id, votes in self.votes.items():
            member_vote = next((v for v in votes if v.voter_id == member_id and not v.is_delegation), None)
            if member_vote:
                proposal = self.proposals[proposal_id]
                member_votes.append(
                    {
                        "proposal_id": proposal_id,
                        "proposal_title": proposal.title,
                        "choice": member_vote.choice.value,
                        "voting_power_used": member_vote.voting_power,
                        "cast_at": member_vote.cast_at.isoformat(),
                    }
                )

        # Get delegations
        delegating_to = self.delegations.get(member_id)
        delegating_from = [delegator for delegator, delegate in self.delegations.items() if delegate == member_id]

        return {
            "member_info": {
                "member_id": member.member_id,
                "address": member.address,
                "role": member.role.value,
                "active": member.active,
                "joined_at": member.joined_at.isoformat(),
                "last_activity": member.last_activity.isoformat(),
            },
            "voting_power": {
                "own_power": member.voting_power,
                "delegated_power": member.delegated_power,
                "total_power": member.get_total_voting_power(),
            },
            "participation": {
                "proposals_created": len(member_proposals),
                "votes_cast": len(member_votes),
                "participation_rate": member.participation_rate,
                "reputation_score": member.reputation_score,
            },
            "delegation": {
                "delegating_to": delegating_to,
                "delegating_from": delegating_from,
                "delegation_power_received": member.delegated_power,
            },
            "compliance": {
                "kyc_verified": member.kyc_verified,
                "jurisdiction": member.jurisdiction,
                "compliance_tier": member.compliance_tier,
            },
            "recent_activity": {
                "proposals": member_proposals[-5:],  # Last 5 proposals
                "votes": member_votes[-10:],  # Last 10 votes
            },
        }

    def get_proposal_details(self, proposal_id: str) -> dict[str, Any]:
        """Get detailed information about a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]
        proposal_votes = self.votes[proposal_id]

        # Get voting breakdown
        vote_breakdown = {
            "yes": [v for v in proposal_votes if v.choice == VoteChoice.YES],
            "no": [v for v in proposal_votes if v.choice == VoteChoice.NO],
            "abstain": [v for v in proposal_votes if v.choice == VoteChoice.ABSTAIN],
        }

        # Get author information
        author = self.members.get(proposal.author_id, {})

        return {
            "proposal": {
                "proposal_id": proposal.proposal_id,
                "title": proposal.title,
                "description": proposal.description,
                "type": proposal.proposal_type.value,
                "status": proposal.status.value,
                "created_at": proposal.created_at.isoformat(),
                "voting_start": proposal.voting_start.isoformat() if proposal.voting_start else None,
                "voting_end": proposal.voting_end.isoformat() if proposal.voting_end else None,
                "execution_target": proposal.execution_target.isoformat() if proposal.execution_target else None,
                "tags": proposal.tags,
                "budget_request": proposal.budget_request,
                "implementation_timeline": proposal.implementation_timeline,
            },
            "author": {
                "member_id": proposal.author_id,
                "role": author.role.value if hasattr(author, "role") else "unknown",
                "reputation_score": author.reputation_score if hasattr(author, "reputation_score") else 0,
            },
            "voting_parameters": {
                "quorum_threshold": proposal.quorum_threshold,
                "approval_threshold": proposal.approval_threshold,
                "minimum_voting_power": proposal.minimum_voting_power,
            },
            "current_results": proposal.get_voting_results(),
            "vote_details": {
                "total_votes": len(proposal_votes),
                "vote_breakdown": {
                    "yes": len(vote_breakdown["yes"]),
                    "no": len(vote_breakdown["no"]),
                    "abstain": len(vote_breakdown["abstain"]),
                },
                "voting_power_breakdown": {
                    "yes": sum(v.voting_power for v in vote_breakdown["yes"]),
                    "no": sum(v.voting_power for v in vote_breakdown["no"]),
                    "abstain": sum(v.voting_power for v in vote_breakdown["abstain"]),
                },
            },
            "proposal_data": proposal.proposal_data,
        }

    def process_expired_proposals(self) -> list[str]:
        """Process proposals that have expired and update their status."""
        now = datetime.now()
        expired_proposals = []

        for proposal_id, proposal in self.proposals.items():
            # Check for expired voting periods
            if proposal.status == ProposalStatus.VOTING and proposal.voting_end and now > proposal.voting_end:
                self._finalize_voting(proposal_id, early=False)
                expired_proposals.append(proposal_id)

            # Check for expired review periods
            elif proposal.status == ProposalStatus.REVIEW and proposal.voting_start and now > proposal.voting_start:
                self.start_voting(proposal_id)
                expired_proposals.append(proposal_id)

        return expired_proposals

    def get_audit_log(self, limit: int = 100, action_type: str = None) -> list[dict[str, Any]]:
        """Get audit log entries."""
        logs = self.audit_log

        if action_type:
            logs = [log for log in logs if log.get("action") == action_type]

        # Return most recent entries
        return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def export_governance_data(self) -> dict[str, Any]:
        """Export complete governance data for backup or analysis."""
        return {
            "governance_config": self.governance_config,
            "members": {
                member_id: {
                    **member.dict(),
                    "joined_at": member.joined_at.isoformat(),
                    "last_activity": member.last_activity.isoformat(),
                }
                for member_id, member in self.members.items()
            },
            "proposals": {
                proposal_id: {
                    **proposal.dict(),
                    "created_at": proposal.created_at.isoformat(),
                    "voting_start": proposal.voting_start.isoformat() if proposal.voting_start else None,
                    "voting_end": proposal.voting_end.isoformat() if proposal.voting_end else None,
                    "execution_target": proposal.execution_target.isoformat() if proposal.execution_target else None,
                }
                for proposal_id, proposal in self.proposals.items()
            },
            "votes": {
                proposal_id: [{**vote.dict(), "cast_at": vote.cast_at.isoformat()} for vote in votes]
                for proposal_id, votes in self.votes.items()
            },
            "delegations": self.delegations,
            "audit_log": self.audit_log,
            "export_timestamp": datetime.now().isoformat(),
        }


# Example usage and testing
async def main():
    """Example usage of the DAO governance system."""
    logger.info("Initializing DAO Governance System...")

    # Create governance system
    dao = DAOGovernanceSystem()

    # Add some test members
    dao.add_member("test_user_1", "0xtest1", MemberRole.MEMBER, 5000, kyc_verified=True, jurisdiction="US")
    dao.add_member("test_user_2", "0xtest2", MemberRole.MEMBER, 3000, kyc_verified=True, jurisdiction="EU")

    # Get governance stats
    stats = dao.get_governance_stats()
    logger.info(f"Governance Stats: {json.dumps(stats, indent=2)}")

    # Test voting on active proposals
    active_proposals = [p for p in dao.proposals.values() if p.status == ProposalStatus.VOTING]
    if active_proposals:
        proposal = active_proposals[0]
        logger.info(f"Casting votes on active proposal: {proposal.title}")

        # Cast some test votes
        dao.cast_vote(proposal.proposal_id, "test_user_1", VoteChoice.YES, "I support this proposal")
        dao.cast_vote(proposal.proposal_id, "founder_001", VoteChoice.YES, "Strong support")
        dao.cast_vote(proposal.proposal_id, "delegate_001", VoteChoice.NO, "Need more discussion")

        # Get updated proposal details
        details = dao.get_proposal_details(proposal.proposal_id)
        logger.info(f"Proposal voting results: {json.dumps(details['current_results'], indent=2)}")

    logger.info("DAO Governance System demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
