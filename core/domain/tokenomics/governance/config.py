"""Configuration for DAO governance system."""

from dataclasses import dataclass


@dataclass
class GovernanceConfig:
    """Configuration for governance parameters."""

    # Quorum requirements (percentage of total supply)
    quorum_threshold: float = 0.1  # 10% of total supply must vote

    # Supermajority requirements (percentage of votes cast)
    supermajority_threshold: float = 0.6  # 60% of votes must be YES

    # Voting period (seconds)
    voting_period: int = 7 * 24 * 60 * 60  # 7 days

    # Minimum voting power to create proposal
    min_proposal_power: int = 1000

    # Grace period after voting ends before enactment (seconds)
    grace_period: int = 24 * 60 * 60  # 1 day

    # Database path for persistence
    db_path: str = "governance.db"

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.quorum_threshold <= 1:
            raise ValueError("quorum_threshold must be between 0 and 1")

        if not 0 <= self.supermajority_threshold <= 1:
            raise ValueError("supermajority_threshold must be between 0 and 1")

        if self.voting_period <= 0:
            raise ValueError("voting_period must be positive")

        if self.min_proposal_power < 0:
            raise ValueError("min_proposal_power must be non-negative")

        if self.grace_period < 0:
            raise ValueError("grace_period must be non-negative")
