# isort: skip_file
import sys
from collections.abc import Sequence

from src.cli.base import run_cli


def train_agent(args) -> int:
    """Train an agent."""
    print(f"Training {args.agent_type} agent: {args.name}")
    return 0


def create_agent(args) -> int:
    """Create a new agent."""
    print(f"Creating {args.agent_type} agent: {args.name}")
    return 0


def list_agents(args) -> int:
    """List all agents."""
    print("Listing all agents...")
    return 0


def delete_agent(args) -> int:
    """Delete an agent."""
    print(f"Deleting agent: {args.name}")
    return 0


def get_status(args) -> int:
    """Get service status."""
    print("Agent Forge service status: Running")
    return 0


def _configure(parser) -> None:
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument(
        "--agent-type",
        choices=["king", "sage", "magi", "base"],
        default="base",
        help="Type of agent to create/train",
    )
    parser.add_argument("--name", help="Agent name")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )


actions = {
    "train": train_agent,
    "create": create_agent,
    "list": list_agents,
    "delete": delete_agent,
    "status": get_status,
}


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point for Agent Forge service."""
    return run_cli("Agent Forge Service", actions, _configure, args)


if __name__ == "__main__":
    sys.exit(main())
