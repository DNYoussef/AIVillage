#!/usr/bin/env python3
"""Unified Entry Point for AIVillage Platform

This is the main entry point for the AIVillage platform, providing a unified
CLI interface for all services and modes of operation.

Usage:
    python main.py --mode MODE --action ACTION [OPTIONS]

    Modes:
        agent-forge: Agent creation and management
        king: KING agent system operations
        rag: Retrieval-augmented generation
        core: Core utilities and configuration

Examples:
        python main.py --mode agent-forge --action train --config config.yaml
        python main.py --mode king --action run --task "analyze data"
        python main.py --mode rag --action query --question "What is AI?"
        python main.py --mode core --action status
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Allow importing experimental agents as top-level 'agents'
sys.path.insert(0, str(Path(__file__).parent / "experimental" / "agents"))


def create_parser():
    """Create the unified argument parser"""
    parser = argparse.ArgumentParser(
        description="AIVillage Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode agent-forge --action train --config config.yaml
  %(prog)s --mode king --action run --task "analyze data"
  %(prog)s --mode rag --action query --question "What is AI?"
  %(prog)s --mode core --action status
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["agent-forge", "king", "rag", "core"],
        required=True,
        help="Service mode to run",
    )

    parser.add_argument(
        "--action", "-a", required=True, help="Action to perform (depends on mode)"
    )

    parser.add_argument("--config", "-c", help="Configuration file path")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Mode-specific arguments
    parser.add_argument("--task", help="Task description (for king mode)")

    parser.add_argument("--question", help="Question to query (for rag mode)")

    parser.add_argument("--document", help="Document to index (for rag mode)")

    parser.add_argument(
        "--agent-type",
        choices=["king", "sage", "magi", "base"],
        default="base",
        help="Type of agent to create/train (for agent-forge mode)",
    )

    parser.add_argument("--name", help="Agent name (for agent-forge mode)")

    parser.add_argument("--input", help="Input file or directory")

    parser.add_argument("--output", help="Output file or directory")

    return parser


def run_agent_forge_mode(args):
    """Run Agent Forge mode"""
    try:
        from agent_forge.main import main as agent_forge_main

        # Convert unified args to agent-forge format
        agent_args = [args.action]
        if args.config:
            agent_args.extend(["--config", args.config])
        if args.agent_type:
            agent_args.extend(["--agent-type", args.agent_type])
        if args.name:
            agent_args.extend(["--name", args.name])
        if args.verbose:
            agent_args.append("--verbose")

        return agent_forge_main(agent_args)
    except ImportError as e:
        print(f"Error: Agent Forge module not found: {e}")
        return 1


def run_king_mode(args):
    """Run KING agent mode"""
    try:
        from agents.king.main import main as king_main
    except Exception:
        import importlib.util

        king_path = (
            Path(__file__).parent
            / "experimental"
            / "agents"
            / "agents"
            / "king"
            / "main.py"
        )
        spec = importlib.util.spec_from_file_location("king_main", king_path)
        king_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(king_module)
        king_main = king_module.main

    # Convert unified args to king format
    king_args = [args.action]
    if args.config:
        king_args.extend(["--config", args.config])
    if args.task:
        king_args.extend(["--task", args.task])
    if args.input:
        king_args.extend(["--input", args.input])
    if args.output:
        king_args.extend(["--output", args.output])
    if args.verbose:
        king_args.append("--verbose")

    return king_main(king_args)


def run_rag_mode(args):
    """Run RAG system mode"""
    try:
        from rag_system.main import main as rag_main

        # Convert unified args to rag format
        rag_args = [args.action]
        if args.config:
            rag_args.extend(["--config", args.config])
        if args.question:
            rag_args.extend(["--question", args.question])
        if args.document:
            rag_args.extend(["--document", args.document])
        if args.input:
            rag_args.extend(["--input", args.input])
        if args.output:
            rag_args.extend(["--output", args.output])
        if args.verbose:
            rag_args.append("--verbose")

        return rag_main(rag_args)
    except ImportError as e:
        print(f"Error: RAG system module not found: {e}")
        return 1


def run_core_mode(args):
    """Run core utilities mode"""
    try:
        from agent_forge.core.main import main as core_main

        # Core service uses click and expects config_file and out_path
        if args.action == "merge":
            if not args.config:
                print("Error: --config is required for core merge action")
                return 1
            if not args.output:
                print("Error: --output is required for core merge action")
                return 1

            return core_main([args.config, args.output])
        print("Core mode supports: merge")
        return 1

    except ImportError as e:
        print(f"Error: Core module not found: {e}")
        return 1


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        print(f"Running in {args.mode} mode with action: {args.action}")

    # Route to appropriate mode handler
    mode_handlers = {
        "agent-forge": run_agent_forge_mode,
        "king": run_king_mode,
        "rag": run_rag_mode,
        "core": run_core_mode,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        return handler(args)
    print(f"Error: Unknown mode '{args.mode}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
