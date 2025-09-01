"""
Contribution Ledger for DAO Rewards

Maintains comprehensive ledger of all fog computing contributions for DAO governance
and reward distribution. Provides transparent, auditable record of participant value.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import aiofiles

logger = logging.getLogger(__name__)


class ContributionType(Enum):
    """Types of contributions to the fog network."""

    COMPUTE_PROVISION = "compute_provision"  # Providing compute resources
    STORAGE_PROVISION = "storage_provision"  # Providing storage
    BANDWIDTH_PROVISION = "bandwidth_provision"  # Providing bandwidth
    HIDDEN_SERVICE_HOST = "hidden_service_host"  # Hosting hidden services
    MIXNET_NODE = "mixnet_node"  # Operating mixnet node
    VALIDATOR = "validator"  # Validating transactions
    DISCOVERY_SEED = "discovery_seed"  # Network discovery seed
    BUG_REPORT = "bug_report"  # Security/bug reporting
    CODE_CONTRIBUTION = "code_contribution"  # Open source contributions
    DOCUMENTATION = "documentation"  # Documentation improvements
    COMMUNITY_SUPPORT = "community_support"  # Helping other users
    ONBOARDING = "onboarding"  # Bringing new participants


class RewardTier(Enum):
    """Reward tiers based on contribution quality and impact."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


class ContributionStatus(Enum):
    """Status of contribution entries."""

    PENDING = "pending"  # Awaiting verification
    VERIFIED = "verified"  # Verified and approved
    DISPUTED = "disputed"  # Under dispute
    REJECTED = "rejected"  # Rejected
    EXPIRED = "expired"  # Expired without verification


@dataclass
class ContributionMetrics:
    """Detailed metrics for a contribution."""

    compute_hours: float = 0.0
    memory_gb_hours: float = 0.0
    storage_gb_hours: float = 0.0
    bandwidth_gb: float = 0.0
    uptime_hours: float = 0.0
    availability_percent: float = 0.0
    response_time_ms: float = 0.0
    success_rate: float = 0.0
    users_served: int = 0
    tasks_completed: int = 0
    data_processed_gb: float = 0.0
    security_incidents: int = 0

    # Quality metrics
    user_satisfaction: float = 0.0
    peer_ratings: list[float] = field(default_factory=list)
    reliability_score: float = 0.0
    innovation_factor: float = 0.0


@dataclass
class ContributionRecord:
    """A single contribution record."""

    contribution_id: str
    contributor_id: str
    contribution_type: ContributionType
    timestamp: datetime
    duration_hours: float
    metrics: ContributionMetrics
    metadata: dict[str, Any] = field(default_factory=dict)

    # Verification
    status: ContributionStatus = ContributionStatus.PENDING
    verified_by: str | None = None
    verification_timestamp: datetime | None = None
    verification_notes: str = ""

    # Rewards
    reward_tier: RewardTier | None = None
    base_reward: float = 0.0
    multiplier: float = 1.0
    final_reward: float = 0.0
    reward_paid: bool = False
    reward_transaction: str | None = None


@dataclass
class ContributorProfile:
    """Profile of a network contributor."""

    contributor_id: str
    join_date: datetime
    total_contributions: int = 0
    active_days: int = 0
    total_rewards: float = 0.0
    current_tier: RewardTier = RewardTier.BRONZE
    reputation_score: float = 0.0

    # Contribution breakdown
    contribution_types: dict[ContributionType, int] = field(default_factory=dict)
    tier_progression: list[tuple[datetime, RewardTier]] = field(default_factory=list)

    # Performance metrics
    average_uptime: float = 0.0
    average_quality: float = 0.0
    consistency_score: float = 0.0
    growth_trajectory: float = 0.0

    # Social metrics
    referrals: int = 0
    mentorship_hours: float = 0.0
    community_impact: float = 0.0


@dataclass
class DAOProposal:
    """DAO governance proposal."""

    proposal_id: str
    title: str
    description: str
    proposer_id: str
    proposal_type: str
    created_at: datetime
    voting_ends_at: datetime

    # Voting results
    votes_for: float = 0.0
    votes_against: float = 0.0
    votes_abstain: float = 0.0
    total_voting_power: float = 0.0

    status: str = "active"  # active, passed, rejected, expired
    execution_date: datetime | None = None

    # Proposal details
    requested_budget: float = 0.0
    implementation_timeline: str = ""
    success_criteria: list[str] = field(default_factory=list)


class ContributionLedger:
    """
    Comprehensive Contribution Ledger for DAO Rewards.

    Maintains transparent, auditable records of all network contributions
    and calculates fair rewards based on impact, quality, and scarcity.
    """

    def __init__(self, data_dir: str = "contribution_ledger"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Ledger storage
        self.contributions: dict[str, ContributionRecord] = {}
        self.contributors: dict[str, ContributorProfile] = {}
        self.proposals: dict[str, DAOProposal] = {}

        # Reward calculation parameters
        self.reward_multipliers = {
            ContributionType.COMPUTE_PROVISION: 1.0,
            ContributionType.STORAGE_PROVISION: 0.8,
            ContributionType.BANDWIDTH_PROVISION: 0.6,
            ContributionType.HIDDEN_SERVICE_HOST: 1.5,
            ContributionType.MIXNET_NODE: 1.2,
            ContributionType.VALIDATOR: 2.0,
            ContributionType.DISCOVERY_SEED: 1.1,
            ContributionType.BUG_REPORT: 3.0,
            ContributionType.CODE_CONTRIBUTION: 4.0,
            ContributionType.DOCUMENTATION: 2.0,
            ContributionType.COMMUNITY_SUPPORT: 1.3,
            ContributionType.ONBOARDING: 1.8,
        }

        self.tier_thresholds = {
            RewardTier.BRONZE: 0,
            RewardTier.SILVER: 100,
            RewardTier.GOLD: 500,
            RewardTier.PLATINUM: 2000,
            RewardTier.DIAMOND: 10000,
        }

        self.tier_multipliers = {
            RewardTier.BRONZE: 1.0,
            RewardTier.SILVER: 1.2,
            RewardTier.GOLD: 1.5,
            RewardTier.PLATINUM: 2.0,
            RewardTier.DIAMOND: 3.0,
        }

        # Analytics
        self.contribution_history: deque = deque(maxlen=100000)
        self.reward_distributions: dict[str, float] = defaultdict(float)
        self.network_stats: dict[str, Any] = defaultdict(int)

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        logger.info("Contribution Ledger initialized")

    async def start(self):
        """Start the contribution ledger system."""
        if self._running:
            return

        logger.info("Starting Contribution Ledger")
        self._running = True

        # Load existing data
        await self._load_ledger_data()

        # Start background tasks
        tasks = [
            self._contribution_validator(),
            self._reward_calculator(),
            self._analytics_updater(),
            self._tier_updater(),
            self._data_persister(),
            self._proposal_manager(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Contribution Ledger started successfully")

    async def stop(self):
        """Stop the contribution ledger system."""
        if not self._running:
            return

        logger.info("Stopping Contribution Ledger")
        self._running = False

        # Save all data
        await self._save_ledger_data()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Contribution Ledger stopped")

    async def record_contribution(
        self,
        contributor_id: str,
        contribution_type: ContributionType,
        duration_hours: float,
        metrics: ContributionMetrics,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Record a new contribution."""
        contribution_id = f"contrib_{int(time.time())}_{uuid4().hex[:8]}"

        # Create contribution record
        record = ContributionRecord(
            contribution_id=contribution_id,
            contributor_id=contributor_id,
            contribution_type=contribution_type,
            timestamp=datetime.now(),
            duration_hours=duration_hours,
            metrics=metrics,
            metadata=metadata or {},
        )

        # Store record
        self.contributions[contribution_id] = record

        # Update contributor profile
        await self._update_contributor_profile(contributor_id, record)

        # Add to history
        self.contribution_history.append(
            {
                "id": contribution_id,
                "contributor": contributor_id,
                "type": contribution_type.value,
                "timestamp": record.timestamp.isoformat(),
                "duration": duration_hours,
            }
        )

        logger.info(f"Recorded contribution {contribution_id} by {contributor_id}")
        return contribution_id

    async def verify_contribution(
        self, contribution_id: str, verifier_id: str, approved: bool, notes: str = ""
    ) -> bool:
        """Verify a contribution."""
        if contribution_id not in self.contributions:
            return False

        record = self.contributions[contribution_id]

        if approved:
            record.status = ContributionStatus.VERIFIED
            # Calculate rewards
            await self._calculate_reward(record)
        else:
            record.status = ContributionStatus.REJECTED

        record.verified_by = verifier_id
        record.verification_timestamp = datetime.now()
        record.verification_notes = notes

        logger.info(f"Contribution {contribution_id} {'approved' if approved else 'rejected'} by {verifier_id}")
        return True

    async def distribute_rewards(self, contribution_ids: list[str]) -> dict[str, float]:
        """Distribute rewards for verified contributions."""
        rewards_distributed = {}

        for contribution_id in contribution_ids:
            if contribution_id not in self.contributions:
                continue

            record = self.contributions[contribution_id]

            # Check if already paid
            if record.reward_paid or record.status != ContributionStatus.VERIFIED:
                continue

            # Distribute reward (mock transaction)
            transaction_id = f"tx_{int(time.time())}_{uuid4().hex[:8]}"

            # Mark as paid
            record.reward_paid = True
            record.reward_transaction = transaction_id

            # Update totals
            if record.contributor_id not in rewards_distributed:
                rewards_distributed[record.contributor_id] = 0.0
            rewards_distributed[record.contributor_id] += record.final_reward

            # Update contributor profile
            contributor = self.contributors[record.contributor_id]
            contributor.total_rewards += record.final_reward

            self.reward_distributions[record.contributor_id] += record.final_reward

        logger.info(f"Distributed rewards to {len(rewards_distributed)} contributors")
        return rewards_distributed

    async def create_dao_proposal(
        self,
        title: str,
        description: str,
        proposer_id: str,
        proposal_type: str,
        voting_period_hours: int = 168,  # 1 week
        requested_budget: float = 0.0,
    ) -> str:
        """Create a new DAO governance proposal."""
        proposal_id = f"prop_{int(time.time())}_{uuid4().hex[:8]}"

        proposal = DAOProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer_id=proposer_id,
            proposal_type=proposal_type,
            created_at=datetime.now(),
            voting_ends_at=datetime.now() + timedelta(hours=voting_period_hours),
            requested_budget=requested_budget,
        )

        self.proposals[proposal_id] = proposal

        logger.info(f"Created DAO proposal {proposal_id}: {title}")
        return proposal_id

    async def vote_on_proposal(
        self, proposal_id: str, voter_id: str, vote: str, voting_power: float  # "for", "against", "abstain"
    ) -> bool:
        """Vote on a DAO proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        # Check if voting is still open
        if datetime.now() > proposal.voting_ends_at:
            return False

        # Record vote
        if vote == "for":
            proposal.votes_for += voting_power
        elif vote == "against":
            proposal.votes_against += voting_power
        elif vote == "abstain":
            proposal.votes_abstain += voting_power
        else:
            return False

        proposal.total_voting_power += voting_power

        logger.info(f"Recorded {vote} vote by {voter_id} on proposal {proposal_id}")
        return True

    async def get_contributor_stats(self, contributor_id: str) -> dict[str, Any] | None:
        """Get comprehensive statistics for a contributor."""
        if contributor_id not in self.contributors:
            return None

        profile = self.contributors[contributor_id]

        # Get recent contributions
        recent_contributions = []
        for record in self.contributions.values():
            if record.contributor_id == contributor_id and record.timestamp > datetime.now() - timedelta(days=30):
                recent_contributions.append(
                    {
                        "id": record.contribution_id,
                        "type": record.contribution_type.value,
                        "timestamp": record.timestamp.isoformat(),
                        "reward": record.final_reward,
                        "status": record.status.value,
                    }
                )

        # Calculate rankings
        all_contributors = list(self.contributors.values())

        # Rank by total rewards
        reward_ranking = sorted(all_contributors, key=lambda c: c.total_rewards, reverse=True)
        reward_rank = next((i + 1 for i, c in enumerate(reward_ranking) if c.contributor_id == contributor_id), None)

        # Rank by reputation
        reputation_ranking = sorted(all_contributors, key=lambda c: c.reputation_score, reverse=True)
        reputation_rank = next(
            (i + 1 for i, c in enumerate(reputation_ranking) if c.contributor_id == contributor_id), None
        )

        return {
            "profile": asdict(profile),
            "recent_contributions": recent_contributions,
            "rankings": {
                "reward_rank": reward_rank,
                "reputation_rank": reputation_rank,
                "total_contributors": len(all_contributors),
            },
            "performance": {
                "contributions_last_30_days": len(recent_contributions),
                "avg_daily_contributions": len(recent_contributions) / 30,
                "total_rewards_last_30_days": sum(c["reward"] for c in recent_contributions),
            },
        }

    async def get_network_analytics(self) -> dict[str, Any]:
        """Get comprehensive network analytics."""
        now = datetime.now()

        # Time-based analytics
        daily_contributions = defaultdict(int)
        monthly_contributions = defaultdict(int)
        contribution_types = defaultdict(int)

        for record in self.contributions.values():
            day_key = record.timestamp.strftime("%Y-%m-%d")
            month_key = record.timestamp.strftime("%Y-%m")

            daily_contributions[day_key] += 1
            monthly_contributions[month_key] += 1
            contribution_types[record.contribution_type.value] += 1

        # Contributor analytics
        active_contributors_30d = set()
        total_rewards_paid = 0.0
        verified_contributions = 0
        pending_contributions = 0

        for record in self.contributions.values():
            if record.timestamp > now - timedelta(days=30):
                active_contributors_30d.add(record.contributor_id)

            if record.status == ContributionStatus.VERIFIED:
                verified_contributions += 1
                if record.reward_paid:
                    total_rewards_paid += record.final_reward
            elif record.status == ContributionStatus.PENDING:
                pending_contributions += 1

        # Tier distribution
        tier_distribution = defaultdict(int)
        for contributor in self.contributors.values():
            tier_distribution[contributor.current_tier.value] += 1

        # DAO governance stats
        active_proposals = sum(1 for p in self.proposals.values() if p.status == "active" and now < p.voting_ends_at)
        total_voting_power = sum(p.total_voting_power for p in self.proposals.values())

        return {
            "overview": {
                "total_contributions": len(self.contributions),
                "total_contributors": len(self.contributors),
                "active_contributors_30d": len(active_contributors_30d),
                "verified_contributions": verified_contributions,
                "pending_contributions": pending_contributions,
                "total_rewards_paid": total_rewards_paid,
            },
            "time_series": {
                "daily_contributions": dict(daily_contributions),
                "monthly_contributions": dict(monthly_contributions),
            },
            "distribution": {
                "contribution_types": dict(contribution_types),
                "contributor_tiers": dict(tier_distribution),
            },
            "governance": {
                "total_proposals": len(self.proposals),
                "active_proposals": active_proposals,
                "total_voting_power": total_voting_power,
            },
        }

    async def get_leaderboard(
        self, metric: str = "total_rewards", limit: int = 100, time_range: str | None = None
    ) -> list[dict[str, Any]]:
        """Get contributor leaderboard."""
        contributors = list(self.contributors.values())

        if time_range:
            # Filter contributions by time range
            cutoff = datetime.now()
            if time_range == "30d":
                cutoff = cutoff - timedelta(days=30)
            elif time_range == "7d":
                cutoff = cutoff - timedelta(days=7)
            elif time_range == "24h":
                cutoff = cutoff - timedelta(hours=24)

        # Sort by metric
        if metric == "total_rewards":
            contributors.sort(key=lambda c: c.total_rewards, reverse=True)
        elif metric == "reputation_score":
            contributors.sort(key=lambda c: c.reputation_score, reverse=True)
        elif metric == "total_contributions":
            contributors.sort(key=lambda c: c.total_contributions, reverse=True)
        elif metric == "consistency_score":
            contributors.sort(key=lambda c: c.consistency_score, reverse=True)

        # Format leaderboard
        leaderboard = []
        for i, contributor in enumerate(contributors[:limit]):
            leaderboard.append(
                {
                    "rank": i + 1,
                    "contributor_id": contributor.contributor_id,
                    "current_tier": contributor.current_tier.value,
                    "total_rewards": contributor.total_rewards,
                    "reputation_score": contributor.reputation_score,
                    "total_contributions": contributor.total_contributions,
                    "join_date": contributor.join_date.isoformat(),
                    "active_days": contributor.active_days,
                }
            )

        return leaderboard

    async def _update_contributor_profile(self, contributor_id: str, record: ContributionRecord):
        """Update contributor profile with new contribution."""
        if contributor_id not in self.contributors:
            # Create new contributor profile
            profile = ContributorProfile(contributor_id=contributor_id, join_date=datetime.now())
            self.contributors[contributor_id] = profile
        else:
            profile = self.contributors[contributor_id]

        # Update counts
        profile.total_contributions += 1

        if record.contribution_type not in profile.contribution_types:
            profile.contribution_types[record.contribution_type] = 0
        profile.contribution_types[record.contribution_type] += 1

        # Update active days
        days_active = (datetime.now() - profile.join_date).days + 1
        profile.active_days = days_active

    async def _calculate_reward(self, record: ContributionRecord):
        """Calculate reward for a contribution."""
        # Get contributor profile
        contributor = self.contributors[record.contributor_id]

        # Base reward calculation
        base_reward = 0.0

        if record.contribution_type == ContributionType.COMPUTE_PROVISION:
            base_reward = (
                record.metrics.compute_hours * 10 + record.metrics.memory_gb_hours * 2 + record.metrics.uptime_hours * 5
            )

        elif record.contribution_type == ContributionType.STORAGE_PROVISION:
            base_reward = record.metrics.storage_gb_hours * 1.5

        elif record.contribution_type == ContributionType.BANDWIDTH_PROVISION:
            base_reward = record.metrics.bandwidth_gb * 0.8

        elif record.contribution_type == ContributionType.HIDDEN_SERVICE_HOST:
            base_reward = record.metrics.uptime_hours * 8 + record.metrics.users_served * 2

        elif record.contribution_type == ContributionType.CODE_CONTRIBUTION:
            base_reward = 500 + record.metrics.innovation_factor * 1000

        else:
            # Default calculation
            base_reward = record.duration_hours * 10

        # Apply type multiplier
        type_multiplier = self.reward_multipliers.get(record.contribution_type, 1.0)

        # Apply quality multipliers
        quality_multiplier = 1.0
        if record.metrics.success_rate > 0.95:
            quality_multiplier += 0.2
        if record.metrics.user_satisfaction > 4.0:
            quality_multiplier += 0.3
        if record.metrics.peer_ratings and sum(record.metrics.peer_ratings) / len(record.metrics.peer_ratings) > 4.0:
            quality_multiplier += 0.25

        # Apply tier multiplier
        tier_multiplier = self.tier_multipliers[contributor.current_tier]

        # Apply scarcity multiplier (more reward for rare contribution types)
        total_of_type = sum(1 for r in self.contributions.values() if r.contribution_type == record.contribution_type)
        scarcity_multiplier = max(1.0, 10.0 / (total_of_type + 1))

        # Calculate final reward
        total_multiplier = type_multiplier * quality_multiplier * tier_multiplier * scarcity_multiplier
        final_reward = base_reward * total_multiplier

        # Determine reward tier based on final reward
        if final_reward >= 5000:
            reward_tier = RewardTier.DIAMOND
        elif final_reward >= 1000:
            reward_tier = RewardTier.PLATINUM
        elif final_reward >= 250:
            reward_tier = RewardTier.GOLD
        elif final_reward >= 50:
            reward_tier = RewardTier.SILVER
        else:
            reward_tier = RewardTier.BRONZE

        # Update record
        record.base_reward = base_reward
        record.multiplier = total_multiplier
        record.final_reward = final_reward
        record.reward_tier = reward_tier

        logger.debug(f"Calculated reward for {record.contribution_id}: {final_reward} FOG tokens")

    async def _contribution_validator(self):
        """Background task to validate pending contributions."""
        while self._running:
            try:
                # Auto-verify low-risk contributions after timeout
                cutoff = datetime.now() - timedelta(hours=72)

                for record in self.contributions.values():
                    if record.status == ContributionStatus.PENDING and record.timestamp < cutoff:
                        # Auto-approve if meets criteria
                        if await self._should_auto_approve(record):
                            await self.verify_contribution(
                                record.contribution_id,
                                "system_auto_verifier",
                                approved=True,
                                notes="Auto-approved after 72h verification period",
                            )

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in contribution validator: {e}")
                await asyncio.sleep(300)

    async def _should_auto_approve(self, record: ContributionRecord) -> bool:
        """Check if contribution should be auto-approved."""
        # Auto-approve compute/storage/bandwidth under certain thresholds
        if record.contribution_type in [
            ContributionType.COMPUTE_PROVISION,
            ContributionType.STORAGE_PROVISION,
            ContributionType.BANDWIDTH_PROVISION,
        ]:
            return record.duration_hours <= 24 and record.metrics.success_rate >= 0.9

        # Always require manual review for high-value contributions
        return False

    async def _reward_calculator(self):
        """Background task to calculate rewards for verified contributions."""
        while self._running:
            try:
                for record in self.contributions.values():
                    if record.status == ContributionStatus.VERIFIED and record.final_reward == 0.0:
                        await self._calculate_reward(record)

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error(f"Error in reward calculator: {e}")
                await asyncio.sleep(300)

    async def _analytics_updater(self):
        """Background task to update network analytics."""
        while self._running:
            try:
                # Update network statistics
                self.network_stats["total_contributions"] = len(self.contributions)
                self.network_stats["total_contributors"] = len(self.contributors)
                self.network_stats["active_proposals"] = sum(
                    1 for p in self.proposals.values() if p.status == "active" and datetime.now() < p.voting_ends_at
                )

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error in analytics updater: {e}")
                await asyncio.sleep(60)

    async def _tier_updater(self):
        """Background task to update contributor tiers."""
        while self._running:
            try:
                for contributor in self.contributors.values():
                    old_tier = contributor.current_tier
                    new_tier = self._calculate_tier(contributor)

                    if new_tier != old_tier:
                        contributor.current_tier = new_tier
                        contributor.tier_progression.append((datetime.now(), new_tier))

                        logger.info(f"Contributor {contributor.contributor_id} promoted to {new_tier.value}")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in tier updater: {e}")
                await asyncio.sleep(300)

    def _calculate_tier(self, contributor: ContributorProfile) -> RewardTier:
        """Calculate appropriate tier for contributor."""
        score = contributor.total_rewards + contributor.reputation_score * 100 + contributor.total_contributions * 10

        if score >= self.tier_thresholds[RewardTier.DIAMOND]:
            return RewardTier.DIAMOND
        elif score >= self.tier_thresholds[RewardTier.PLATINUM]:
            return RewardTier.PLATINUM
        elif score >= self.tier_thresholds[RewardTier.GOLD]:
            return RewardTier.GOLD
        elif score >= self.tier_thresholds[RewardTier.SILVER]:
            return RewardTier.SILVER
        else:
            return RewardTier.BRONZE

    async def _data_persister(self):
        """Background task to persist data to disk."""
        while self._running:
            try:
                await self._save_ledger_data()
                await asyncio.sleep(1800)  # Save every 30 minutes

            except Exception as e:
                logger.error(f"Error in data persister: {e}")
                await asyncio.sleep(300)

    async def _proposal_manager(self):
        """Background task to manage DAO proposals."""
        while self._running:
            try:
                now = datetime.now()

                for proposal in self.proposals.values():
                    if proposal.status == "active" and now > proposal.voting_ends_at:
                        # Close voting and determine outcome
                        total_votes = proposal.votes_for + proposal.votes_against + proposal.votes_abstain

                        if total_votes > 0 and proposal.votes_for > proposal.votes_against:
                            proposal.status = "passed"
                            logger.info(f"DAO proposal {proposal.proposal_id} passed")
                        else:
                            proposal.status = "rejected"
                            logger.info(f"DAO proposal {proposal.proposal_id} rejected")

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in proposal manager: {e}")
                await asyncio.sleep(300)

    async def _save_ledger_data(self):
        """Save ledger data to disk."""
        # Save contributions
        contributions_file = self.data_dir / "contributions.json"
        contributions_data = {}
        for contrib_id, record in self.contributions.items():
            data = asdict(record)
            data["timestamp"] = record.timestamp.isoformat()
            if record.verification_timestamp:
                data["verification_timestamp"] = record.verification_timestamp.isoformat()
            data["contribution_type"] = record.contribution_type.value
            data["status"] = record.status.value
            if record.reward_tier:
                data["reward_tier"] = record.reward_tier.value
            contributions_data[contrib_id] = data

        async with aiofiles.open(contributions_file, "w") as f:
            await f.write(json.dumps(contributions_data, indent=2))

        # Save contributors
        contributors_file = self.data_dir / "contributors.json"
        contributors_data = {}
        for contrib_id, profile in self.contributors.items():
            data = asdict(profile)
            data["join_date"] = profile.join_date.isoformat()
            data["current_tier"] = profile.current_tier.value
            data["contribution_types"] = {k.value: v for k, v in profile.contribution_types.items()}
            data["tier_progression"] = [(dt.isoformat(), tier.value) for dt, tier in profile.tier_progression]
            contributors_data[contrib_id] = data

        async with aiofiles.open(contributors_file, "w") as f:
            await f.write(json.dumps(contributors_data, indent=2))

        # Save proposals
        proposals_file = self.data_dir / "proposals.json"
        proposals_data = {}
        for prop_id, proposal in self.proposals.items():
            data = asdict(proposal)
            data["created_at"] = proposal.created_at.isoformat()
            data["voting_ends_at"] = proposal.voting_ends_at.isoformat()
            if proposal.execution_date:
                data["execution_date"] = proposal.execution_date.isoformat()
            proposals_data[prop_id] = data

        async with aiofiles.open(proposals_file, "w") as f:
            await f.write(json.dumps(proposals_data, indent=2))

    async def _load_ledger_data(self):
        """Load ledger data from disk."""
        try:
            # Load contributions
            contributions_file = self.data_dir / "contributions.json"
            if contributions_file.exists():
                async with aiofiles.open(contributions_file, "r") as f:
                    contributions_data = json.loads(await f.read())

                for contrib_id, data in contributions_data.items():
                    # Convert back to proper types
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    if data.get("verification_timestamp"):
                        data["verification_timestamp"] = datetime.fromisoformat(data["verification_timestamp"])
                    data["contribution_type"] = ContributionType(data["contribution_type"])
                    data["status"] = ContributionStatus(data["status"])
                    if data.get("reward_tier"):
                        data["reward_tier"] = RewardTier(data["reward_tier"])

                    # Create metrics object
                    metrics_data = data.pop("metrics")
                    metrics = ContributionMetrics(**metrics_data)
                    data["metrics"] = metrics

                    record = ContributionRecord(**data)
                    self.contributions[contrib_id] = record

            # Load contributors
            contributors_file = self.data_dir / "contributors.json"
            if contributors_file.exists():
                async with aiofiles.open(contributors_file, "r") as f:
                    contributors_data = json.loads(await f.read())

                for contrib_id, data in contributors_data.items():
                    data["join_date"] = datetime.fromisoformat(data["join_date"])
                    data["current_tier"] = RewardTier(data["current_tier"])
                    data["contribution_types"] = {ContributionType(k): v for k, v in data["contribution_types"].items()}
                    data["tier_progression"] = [
                        (datetime.fromisoformat(dt), RewardTier(tier)) for dt, tier in data["tier_progression"]
                    ]

                    profile = ContributorProfile(**data)
                    self.contributors[contrib_id] = profile

            # Load proposals
            proposals_file = self.data_dir / "proposals.json"
            if proposals_file.exists():
                async with aiofiles.open(proposals_file, "r") as f:
                    proposals_data = json.loads(await f.read())

                for prop_id, data in proposals_data.items():
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    data["voting_ends_at"] = datetime.fromisoformat(data["voting_ends_at"])
                    if data.get("execution_date"):
                        data["execution_date"] = datetime.fromisoformat(data["execution_date"])

                    proposal = DAOProposal(**data)
                    self.proposals[prop_id] = proposal

            logger.info(
                f"Loaded {len(self.contributions)} contributions, {len(self.contributors)} contributors, {len(self.proposals)} proposals"
            )

        except Exception as e:
            logger.error(f"Error loading ledger data: {e}")
            # Continue with empty state
