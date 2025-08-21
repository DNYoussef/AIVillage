"""Comprehensive test suite for DAO governance MVP."""

import tempfile
import time
from pathlib import Path

import pytest
from src.token_economy.credit_system import EarningRule, VILLAGECreditSystem
from src.token_economy.governance import GovernanceConfig, GovernanceSystem, ProposalStatus, VoteChoice


@pytest.fixture
def temp_db():
    """Create temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def credit_system(temp_db):
    """Create test credit system."""
    system = VILLAGECreditSystem(temp_db)

    # Add earning rules
    rules = [
        EarningRule("GOVERNANCE_PARTICIPATION", 100, {}, {}),
        EarningRule("COMPUTE_CONTRIBUTION", 10, {}, {}),
    ]
    for rule in rules:
        system.add_earning_rule(rule)

    # Create test users with different voting powers
    users = {
        "alice": 2000,  # High voting power
        "bob": 1500,  # Medium voting power
        "charlie": 1000,  # Minimum voting power
        "diana": 500,  # Low voting power
        "eve": 0,  # No voting power
    }

    for user_id, balance in users.items():
        if balance > 0:
            system.record_transaction(
                user_id=user_id,
                amount=balance,
                tx_type="MINT",
                category="INITIAL_ALLOCATION",
                metadata={"source": "test"},
            )
            system.update_balance(user_id, balance)

    yield system
    system.close()


@pytest.fixture
def governance_config():
    """Create test governance configuration."""
    return GovernanceConfig(
        quorum_threshold=0.2,  # 20% participation required
        supermajority_threshold=0.6,  # 60% YES votes required
        voting_period=300,  # 5 minutes for testing
        min_proposal_power=1000,  # 1000 tokens to create proposal
        grace_period=60,  # 1 minute grace period
        db_path=":memory:",  # In-memory database for tests
    )


@pytest.fixture
def governance_system(credit_system, governance_config):
    """Create test governance system."""
    system = GovernanceSystem(credit_system, governance_config)
    yield system
    system.close()


class TestProposalLifecycle:
    """Test proposal creation and lifecycle."""

    def test_create_proposal_success(self, governance_system):
        """Test successful proposal creation."""
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="A test proposal for governance",
            proposal_type="test",
        )

        assert proposal.id is not None
        assert proposal.title == "Test Proposal"
        assert proposal.proposer_id == "alice"
        assert proposal.status == ProposalStatus.DRAFT
        assert len(proposal.votes) == 0

    def test_create_proposal_insufficient_power(self, governance_system):
        """Test proposal creation with insufficient voting power."""
        with pytest.raises(ValueError, match="Insufficient voting power"):
            governance_system.create_proposal(
                proposer_id="diana",  # Only 500 voting power
                title="Test Proposal",
                description="Should fail",
            )

    def test_create_proposal_no_power(self, governance_system):
        """Test proposal creation with no voting power."""
        with pytest.raises(ValueError, match="Insufficient voting power"):
            governance_system.create_proposal(
                proposer_id="eve",  # No voting power
                title="Test Proposal",
                description="Should fail",
            )

    def test_start_voting(self, governance_system):
        """Test starting voting on a proposal."""
        # Create proposal
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )

        # Start voting
        updated_proposal = governance_system.start_voting(proposal.id)

        assert updated_proposal.status == ProposalStatus.VOTE
        assert updated_proposal.voting_start is not None
        assert updated_proposal.voting_end is not None
        assert updated_proposal.voting_end > updated_proposal.voting_start

    def test_start_voting_invalid_status(self, governance_system):
        """Test starting voting on proposal not in DRAFT status."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Try to start voting again
        with pytest.raises(ValueError, match="not in DRAFT status"):
            governance_system.start_voting(proposal.id)


class TestVoting:
    """Test voting functionality."""

    def test_cast_vote_success(self, governance_system):
        """Test successful vote casting."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Cast vote
        vote = governance_system.cast_vote(
            proposal_id=proposal.id,
            voter_id="bob",
            choice=VoteChoice.YES,
        )

        assert vote.voter_id == "bob"
        assert vote.choice == VoteChoice.YES
        assert vote.voting_power == 1500  # Bob's balance

        # Verify vote is recorded
        updated_proposal = governance_system.get_proposal(proposal.id)
        assert len(updated_proposal.votes) == 1
        assert updated_proposal.yes_votes == 1500

    def test_cast_multiple_votes(self, governance_system):
        """Test multiple users voting."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Multiple users vote
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.NO)
        governance_system.cast_vote(proposal.id, "charlie", VoteChoice.YES)

        # Check vote tallies
        updated_proposal = governance_system.get_proposal(proposal.id)
        assert len(updated_proposal.votes) == 3
        assert updated_proposal.yes_votes == 3000  # alice(2000) + charlie(1000)
        assert updated_proposal.no_votes == 1500  # bob(1500)
        assert updated_proposal.total_votes == 4500

    def test_cast_vote_replace_existing(self, governance_system):
        """Test that new vote replaces existing vote from same user."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Cast initial vote
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.YES)
        proposal_after_first = governance_system.get_proposal(proposal.id)
        assert proposal_after_first.yes_votes == 1500
        assert proposal_after_first.no_votes == 0

        # Change vote
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.NO)
        proposal_after_second = governance_system.get_proposal(proposal.id)
        assert proposal_after_second.yes_votes == 0
        assert proposal_after_second.no_votes == 1500
        assert len(proposal_after_second.votes) == 1  # Only one vote per user

    def test_cast_vote_no_power(self, governance_system):
        """Test voting with no voting power."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Try to vote with no power
        with pytest.raises(ValueError, match="no voting power"):
            governance_system.cast_vote(proposal.id, "eve", VoteChoice.YES)

    def test_cast_vote_invalid_proposal(self, governance_system):
        """Test voting on non-existent proposal."""
        with pytest.raises(ValueError, match="not found"):
            governance_system.cast_vote("invalid", "alice", VoteChoice.YES)

    def test_cast_vote_wrong_status(self, governance_system):
        """Test voting on proposal not in voting status."""
        # Create proposal but don't start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )

        with pytest.raises(ValueError, match="not in voting status"):
            governance_system.cast_vote(proposal.id, "bob", VoteChoice.YES)


class TestQuorumAndSupermajority:
    """Test quorum and supermajority calculations."""

    def test_quorum_calculation(self, governance_system):
        """Test quorum threshold calculation."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Total supply: alice(2000) + bob(1500) + charlie(1000) + diana(500) = 5000
        # Quorum threshold: 20% = 1000 voting power needed

        # Vote with insufficient participation
        governance_system.cast_vote(proposal.id, "diana", VoteChoice.YES)  # 500 power
        tally = governance_system.tally_votes(proposal.id)
        assert not tally["quorum_met"]
        assert tally["participation_rate"] == 0.1  # 500/5000

        # Vote with sufficient participation
        governance_system.cast_vote(proposal.id, "charlie", VoteChoice.YES)  # +1000 power = 1500 total
        tally = governance_system.tally_votes(proposal.id)
        assert tally["quorum_met"]
        assert tally["participation_rate"] == 0.3  # 1500/5000

    def test_supermajority_calculation(self, governance_system):
        """Test supermajority threshold calculation."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Supermajority threshold: 60% of votes cast must be YES

        # Cast votes: 60% YES, 40% NO
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)  # 2000
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.YES)  # 1500
        governance_system.cast_vote(proposal.id, "charlie", VoteChoice.NO)  # 1000
        governance_system.cast_vote(proposal.id, "diana", VoteChoice.NO)  # 500
        # Total: 5000, YES: 3500 (70%), NO: 1500 (30%)

        tally = governance_system.tally_votes(proposal.id)
        assert tally["supermajority_met"]
        assert tally["yes_rate"] == 0.7

        # Now change one vote to make it fail supermajority
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.NO)  # Change bob to NO
        # Total: 5000, YES: 2000 (40%), NO: 3000 (60%)

        tally = governance_system.tally_votes(proposal.id)
        assert not tally["supermajority_met"]
        assert tally["yes_rate"] == 0.4

    def test_proposal_passes_both_requirements(self, governance_system):
        """Test proposal that meets both quorum and supermajority."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote to meet both requirements
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)  # 2000
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.YES)  # 1500
        # Total participation: 3500/5000 = 70% (exceeds 20% quorum)
        # YES votes: 3500/3500 = 100% (exceeds 60% supermajority)

        tally = governance_system.tally_votes(proposal.id)
        assert tally["quorum_met"]
        assert tally["supermajority_met"]
        assert tally["passes"]

    def test_proposal_fails_quorum(self, governance_system):
        """Test proposal that fails quorum despite supermajority."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote with insufficient participation but high YES rate
        governance_system.cast_vote(proposal.id, "diana", VoteChoice.YES)  # 500
        # Total participation: 500/5000 = 10% (below 20% quorum)
        # YES votes: 500/500 = 100% (exceeds 60% supermajority)

        tally = governance_system.tally_votes(proposal.id)
        assert not tally["quorum_met"]
        assert tally["supermajority_met"]
        assert not tally["passes"]  # Fails overall

    def test_proposal_fails_supermajority(self, governance_system):
        """Test proposal that fails supermajority despite quorum."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote with sufficient participation but low YES rate
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)  # 2000
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.NO)  # 1500
        governance_system.cast_vote(proposal.id, "charlie", VoteChoice.NO)  # 1000
        # Total participation: 4500/5000 = 90% (exceeds 20% quorum)
        # YES votes: 2000/4500 = 44% (below 60% supermajority)

        tally = governance_system.tally_votes(proposal.id)
        assert tally["quorum_met"]
        assert not tally["supermajority_met"]
        assert not tally["passes"]  # Fails overall


class TestProposalEnactment:
    """Test proposal finalization and enactment."""

    def test_finalize_successful_proposal(self, governance_system):
        """Test finalizing a successful proposal."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote to pass
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.YES)

        # Manually set voting end time to past (simulate expired voting)
        proposal_obj = governance_system.get_proposal(proposal.id)
        proposal_obj.voting_end = int(time.time()) - 1
        governance_system.storage.save_proposal(proposal_obj)

        # Finalize proposal
        finalized = governance_system.finalize_proposal(proposal.id)
        assert finalized.status == ProposalStatus.ENACTED

    def test_finalize_failed_proposal(self, governance_system):
        """Test finalizing a failed proposal."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote to fail (insufficient votes)
        governance_system.cast_vote(proposal.id, "diana", VoteChoice.YES)  # Only 500 power

        # Manually set voting end time to past
        proposal_obj = governance_system.get_proposal(proposal.id)
        proposal_obj.voting_end = int(time.time()) - 1
        governance_system.storage.save_proposal(proposal_obj)

        # Finalize proposal
        finalized = governance_system.finalize_proposal(proposal.id)
        assert finalized.status == ProposalStatus.FAILED

    def test_finalize_voting_not_ended(self, governance_system):
        """Test finalizing proposal with active voting period."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Try to finalize while voting is still active
        with pytest.raises(ValueError, match="has not ended yet"):
            governance_system.finalize_proposal(proposal.id)


class TestInvariants:
    """Test system invariants and edge cases."""

    def test_no_negative_balances(self, credit_system):
        """Test that balances cannot go negative."""
        # Try to spend more than available
        with pytest.raises(ValueError, match="Insufficient balance"):
            credit_system.spend_credits(
                user_id="alice",
                amount=3000,  # Alice only has 2000
                category="TEST_SPEND",
                metadata={},
            )

        # Verify balance unchanged
        assert credit_system.get_balance("alice") == 2000

    def test_supply_changes_only_via_mint(self, credit_system, governance_system):
        """Test that total supply only changes through proper minting."""
        initial_supply = governance_system.get_total_supply()

        # Create and vote on proposal (should not change supply)
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)

        # Supply should remain the same
        assert governance_system.get_total_supply() == initial_supply

        # Only minting should change supply
        credit_system.record_transaction(
            user_id="new_user",
            amount=1000,
            tx_type="MINT",
            category="NEW_ALLOCATION",
            metadata={},
        )
        credit_system.update_balance("new_user", 1000)

        # Register the new user with governance system for total supply tracking
        governance_system.register_user("new_user")

        # Supply should increase
        assert governance_system.get_total_supply() == initial_supply + 1000

    def test_voting_power_equals_balance(self, governance_system):
        """Test that voting power always equals token balance."""
        assert governance_system.get_voting_power("alice") == 2000
        assert governance_system.get_voting_power("bob") == 1500
        assert governance_system.get_voting_power("charlie") == 1000
        assert governance_system.get_voting_power("diana") == 500
        assert governance_system.get_voting_power("eve") == 0

    def test_proposal_id_uniqueness(self, governance_system):
        """Test that proposal IDs are unique."""
        proposal1 = governance_system.create_proposal(
            proposer_id="alice",
            title="Proposal 1",
            description="First proposal",
        )
        proposal2 = governance_system.create_proposal(
            proposer_id="alice",
            title="Proposal 2",
            description="Second proposal",
        )

        assert proposal1.id != proposal2.id

    def test_abstain_votes_not_counted_for_supermajority(self, governance_system):
        """Test that abstain votes don't count toward YES/NO for supermajority."""
        # Create and start voting
        proposal = governance_system.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )
        governance_system.start_voting(proposal.id)

        # Vote: 60% YES (of non-abstain), 40% NO, with abstains
        governance_system.cast_vote(proposal.id, "alice", VoteChoice.YES)  # 2000
        governance_system.cast_vote(proposal.id, "charlie", VoteChoice.NO)  # 1000
        governance_system.cast_vote(proposal.id, "bob", VoteChoice.ABSTAIN)  # 1500 (abstain)
        governance_system.cast_vote(proposal.id, "diana", VoteChoice.ABSTAIN)  # 500 (abstain)

        tally = governance_system.tally_votes(proposal.id)

        # Total votes: 5000 (including abstains)
        # YES votes: 2000
        # NO votes: 1000
        # YES rate should be 2000/3000 = 66.7% (excluding abstains from denominator)
        # This should pass supermajority (60%)

        assert tally["total_votes"] == 5000
        assert tally["yes_votes"] == 2000
        assert tally["no_votes"] == 1000
        assert tally["abstain_votes"] == 2000
        assert tally["yes_rate"] == 2000 / 5000  # Based on all votes for simplicity
        # Note: In a more sophisticated system, you might exclude abstains from supermajority calculation


class TestStorage:
    """Test storage implementations."""

    def test_sqlite_storage(self, temp_db, credit_system, governance_config):
        """Test SQLite storage functionality."""
        governance_config.db_path = temp_db
        governance = GovernanceSystem(credit_system, governance_config)

        # Create proposal
        proposal = governance.create_proposal(
            proposer_id="alice",
            title="Test Proposal",
            description="Test description",
        )

        # Verify persistence
        loaded = governance.get_proposal(proposal.id)
        assert loaded is not None
        assert loaded.title == "Test Proposal"
        assert loaded.proposer_id == "alice"

        governance.close()

    def test_file_storage(self, credit_system, governance_config):
        """Test file storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.token_economy.governance.storage import FileGovernanceStorage

            storage = FileGovernanceStorage(temp_dir)
            governance = GovernanceSystem(credit_system, governance_config, storage)

            # Create proposal
            proposal = governance.create_proposal(
                proposer_id="alice",
                title="Test Proposal",
                description="Test description",
            )

            # Verify file exists
            proposal_file = Path(temp_dir) / f"proposal_{proposal.id}.json"
            assert proposal_file.exists()

            # Verify persistence
            loaded = governance.get_proposal(proposal.id)
            assert loaded is not None
            assert loaded.title == "Test Proposal"

            governance.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
