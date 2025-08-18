"""CLI interface for DAO governance operations."""

import argparse
import sys

from ..credit_system import EarningRule, VILLAGECreditSystem
from .config import GovernanceConfig
from .governance_system import GovernanceSystem
from .models import ProposalStatus, VoteChoice


def setup_test_credit_system() -> VILLAGECreditSystem:
    """Setup test credit system with some users and balances."""
    credit_system = VILLAGECreditSystem("test_governance_credits.db")

    # Add some earning rules
    rules = [
        EarningRule("GOVERNANCE_PARTICIPATION", 100, {}, {}),
        EarningRule("COMPUTE_CONTRIBUTION", 10, {}, {}),
    ]
    for rule in rules:
        credit_system.add_earning_rule(rule)

    # Give test users some credits
    test_users = ["alice", "bob", "charlie", "diana", "eve"]
    for i, user in enumerate(test_users):
        credits = 1000 + (i * 500)  # Give varying amounts
        credit_system.record_transaction(
            user_id=user,
            amount=credits,
            tx_type="MINT",
            category="INITIAL_ALLOCATION",
            metadata={"source": "test_setup"},
        )
        credit_system.update_balance(user, credits)

    return credit_system


def create_proposal(gov: GovernanceSystem, args) -> None:
    """Create a new proposal."""
    try:
        proposal = gov.create_proposal(
            proposer_id=args.proposer,
            title=args.title,
            description=args.description,
            proposal_type=args.type or "general",
        )
        print(f"Proposal created: {proposal.id}")
        print(f"   Title: {proposal.title}")
        print(f"   Status: {proposal.status.value}")
        print(f"   Proposer: {proposal.proposer_id}")
    except Exception as e:
        print(f"Error creating proposal: {e}")
        sys.exit(1)


def start_voting(gov: GovernanceSystem, args) -> None:
    """Start voting on a proposal."""
    try:
        proposal = gov.start_voting(args.proposal_id)
        print(f"Voting started for proposal: {proposal.id}")
        print(f"   Voting period: {gov.config.voting_period} seconds")
        print(f"   Voting ends at: {proposal.voting_end}")
    except Exception as e:
        print(f"Error starting voting: {e}")
        sys.exit(1)


def cast_vote(gov: GovernanceSystem, args) -> None:
    """Cast a vote on a proposal."""
    try:
        choice = VoteChoice(args.choice.lower())
        vote = gov.cast_vote(
            proposal_id=args.proposal_id,
            voter_id=args.voter,
            choice=choice,
        )
        print(f"SUCCESS: Vote cast by {vote.voter_id}")
        print(f"   Choice: {vote.choice.value}")
        print(f"   Voting power: {vote.voting_power}")
    except Exception as e:
        print(f"ERROR: Error casting vote: {e}")
        sys.exit(1)


def tally_votes(gov: GovernanceSystem, args) -> None:
    """Tally votes for a proposal."""
    try:
        tally = gov.tally_votes(args.proposal_id)
        print(f"RESULTS: Vote tally for proposal: {args.proposal_id}")
        print(f"   Total votes: {tally['total_votes']}")
        print(f"   YES votes: {tally['yes_votes']}")
        print(f"   NO votes: {tally['no_votes']}")
        print(f"   ABSTAIN votes: {tally['abstain_votes']}")
        print(f"   Total supply: {tally['total_supply']}")

        if "participation_rate" in tally:
            print(f"   Participation rate: {tally['participation_rate']:.2%}")
        if "yes_rate" in tally:
            print(f"   YES rate: {tally['yes_rate']:.2%}")

        print(f"   Quorum met: {'SUCCESS:' if tally['quorum_met'] else 'ERROR:'}")
        print(f"   Supermajority met: {'SUCCESS:' if tally['supermajority_met'] else 'ERROR:'}")
        print(f"   Proposal passes: {'SUCCESS:' if tally['passes'] else 'ERROR:'}")
    except Exception as e:
        print(f"ERROR: Error tallying votes: {e}")
        sys.exit(1)


def finalize_proposal(gov: GovernanceSystem, args) -> None:
    """Finalize a proposal after voting."""
    try:
        proposal = gov.finalize_proposal(args.proposal_id)
        print(f"SUCCESS: Proposal finalized: {proposal.id}")
        print(f"   Final status: {proposal.status.value}")

        if proposal.status == ProposalStatus.ENACTED:
            print("   PASSED: Proposal was ENACTED!")
        else:
            print("   ERROR: Proposal FAILED")
    except Exception as e:
        print(f"ERROR: Error finalizing proposal: {e}")
        sys.exit(1)


def list_proposals(gov: GovernanceSystem, args) -> None:
    """List all proposals."""
    status_filter = ProposalStatus(args.status) if args.status else None
    proposals = gov.list_proposals(status_filter)

    if not proposals:
        print("No proposals found.")
        return

    print(f"LIST: Found {len(proposals)} proposal(s):")
    for proposal in proposals:
        print(f"  • {proposal.id[:8]}... - {proposal.title}")
        print(f"    Status: {proposal.status.value}")
        print(f"    Proposer: {proposal.proposer_id}")
        print(f"    Votes: {len(proposal.votes)}")
        print()


def show_proposal(gov: GovernanceSystem, args) -> None:
    """Show detailed proposal information."""
    proposal = gov.get_proposal(args.proposal_id)
    if not proposal:
        print(f"ERROR: Proposal {args.proposal_id} not found")
        sys.exit(1)

    print(f"LIST: Proposal: {proposal.id}")
    print(f"   Title: {proposal.title}")
    print(f"   Description: {proposal.description}")
    print(f"   Proposer: {proposal.proposer_id}")
    print(f"   Status: {proposal.status.value}")
    print(f"   Created: {proposal.created_at}")

    if proposal.voting_start:
        print(f"   Voting start: {proposal.voting_start}")
    if proposal.voting_end:
        print(f"   Voting end: {proposal.voting_end}")

    print(f"   Votes cast: {len(proposal.votes)}")
    if proposal.votes:
        print("   Vote breakdown:")
        for vote in proposal.votes:
            print(f"     {vote.voter_id}: {vote.choice.value} ({vote.voting_power} power)")


def show_balance(gov: GovernanceSystem, args) -> None:
    """Show user's token balance and voting power."""
    balance = gov.get_voting_power(args.user)
    print(f"BALANCE: User: {args.user}")
    print(f"   Balance/Voting Power: {balance}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DAO Governance CLI")
    parser.add_argument("--db", default="governance.db", help="Governance database path")
    parser.add_argument("--setup-test", action="store_true", help="Setup test environment")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create proposal
    create_parser = subparsers.add_parser("create", help="Create a new proposal")
    create_parser.add_argument("proposer", help="Proposer user ID")
    create_parser.add_argument("title", help="Proposal title")
    create_parser.add_argument("description", help="Proposal description")
    create_parser.add_argument("--type", help="Proposal type")

    # Start voting
    vote_start_parser = subparsers.add_parser("start-voting", help="Start voting on proposal")
    vote_start_parser.add_argument("proposal_id", help="Proposal ID")

    # Cast vote
    vote_parser = subparsers.add_parser("vote", help="Cast a vote")
    vote_parser.add_argument("proposal_id", help="Proposal ID")
    vote_parser.add_argument("voter", help="Voter user ID")
    vote_parser.add_argument("choice", choices=["yes", "no", "abstain"], help="Vote choice")

    # Tally votes
    tally_parser = subparsers.add_parser("tally", help="Tally votes for proposal")
    tally_parser.add_argument("proposal_id", help="Proposal ID")

    # Finalize proposal
    finalize_parser = subparsers.add_parser("finalize", help="Finalize proposal")
    finalize_parser.add_argument("proposal_id", help="Proposal ID")

    # List proposals
    list_parser = subparsers.add_parser("list", help="List proposals")
    list_parser.add_argument(
        "--status",
        choices=["draft", "vote", "enacted", "failed"],
        help="Filter by status",
    )

    # Show proposal
    show_parser = subparsers.add_parser("show", help="Show proposal details")
    show_parser.add_argument("proposal_id", help="Proposal ID")

    # Show balance
    balance_parser = subparsers.add_parser("balance", help="Show user balance")
    balance_parser.add_argument("user", help="User ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup systems
    if args.setup_test:
        credit_system = setup_test_credit_system()
        print("Test credit system setup with test users: alice, bob, charlie, diana, eve")
    else:
        credit_system = VILLAGECreditSystem("village_credits.db")

    config = GovernanceConfig(db_path=args.db)
    gov = GovernanceSystem(credit_system, config)

    # Dispatch commands
    try:
        if args.command == "create":
            create_proposal(gov, args)
        elif args.command == "start-voting":
            start_voting(gov, args)
        elif args.command == "vote":
            cast_vote(gov, args)
        elif args.command == "tally":
            tally_votes(gov, args)
        elif args.command == "finalize":
            finalize_proposal(gov, args)
        elif args.command == "list":
            list_proposals(gov, args)
        elif args.command == "show":
            show_proposal(gov, args)
        elif args.command == "balance":
            show_balance(gov, args)
    finally:
        gov.close()
        credit_system.close()


if __name__ == "__main__":
    main()
