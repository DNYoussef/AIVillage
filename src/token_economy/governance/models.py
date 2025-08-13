"""Core data models for DAO governance system."""

from dataclasses import dataclass, field
from enum import Enum
import time


class ProposalStatus(Enum):
    """Proposal lifecycle states."""

    DRAFT = "draft"
    VOTE = "vote"
    ENACTED = "enacted"
    FAILED = "failed"


class VoteChoice(Enum):
    """Voting choices."""

    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """Individual vote record."""

    voter_id: str
    choice: VoteChoice
    voting_power: int
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "voter_id": self.voter_id,
            "choice": self.choice.value,
            "voting_power": self.voting_power,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vote":
        """Create from dictionary."""
        return cls(
            voter_id=data["voter_id"],
            choice=VoteChoice(data["choice"]),
            voting_power=data["voting_power"],
            timestamp=data["timestamp"],
        )


@dataclass
class Proposal:
    """DAO governance proposal."""

    id: str
    title: str
    description: str
    proposer_id: str
    status: ProposalStatus = ProposalStatus.DRAFT
    votes: list[Vote] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    voting_start: int | None = None
    voting_end: int | None = None
    execution_metadata: dict = field(default_factory=dict)

    @property
    def total_votes(self) -> int:
        """Total voting power cast."""
        return sum(vote.voting_power for vote in self.votes)

    @property
    def yes_votes(self) -> int:
        """Total YES voting power."""
        return sum(
            vote.voting_power for vote in self.votes if vote.choice == VoteChoice.YES
        )

    @property
    def no_votes(self) -> int:
        """Total NO voting power."""
        return sum(
            vote.voting_power for vote in self.votes if vote.choice == VoteChoice.NO
        )

    @property
    def abstain_votes(self) -> int:
        """Total ABSTAIN voting power."""
        return sum(
            vote.voting_power
            for vote in self.votes
            if vote.choice == VoteChoice.ABSTAIN
        )

    def get_vote_by_user(self, voter_id: str) -> Vote | None:
        """Get vote by specific user."""
        for vote in self.votes:
            if vote.voter_id == voter_id:
                return vote
        return None

    def add_vote(self, vote: Vote) -> None:
        """Add or update a vote."""
        # Remove existing vote from same user
        self.votes = [v for v in self.votes if v.voter_id != vote.voter_id]
        self.votes.append(vote)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "proposer_id": self.proposer_id,
            "status": self.status.value,
            "votes": [vote.to_dict() for vote in self.votes],
            "created_at": self.created_at,
            "voting_start": self.voting_start,
            "voting_end": self.voting_end,
            "execution_metadata": self.execution_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        """Create from dictionary."""
        votes = [Vote.from_dict(vote_data) for vote_data in data.get("votes", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            proposer_id=data["proposer_id"],
            status=ProposalStatus(data["status"]),
            votes=votes,
            created_at=data["created_at"],
            voting_start=data.get("voting_start"),
            voting_end=data.get("voting_end"),
            execution_metadata=data.get("execution_metadata", {}),
        )
