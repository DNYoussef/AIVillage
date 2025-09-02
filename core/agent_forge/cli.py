#!/usr/bin/env python3
"""
Agent Forge CLI Entry Point

This module provides command-line interface for the Agent Forge system.
"""

import argparse
import sys


def main():
    """Main CLI entry point for Agent Forge"""
    parser = argparse.ArgumentParser(description="Agent Forge - AI Agent Training and Management System")

    parser.add_argument("--version", action="version", version="Agent Forge 2.0.0")

    parser.add_argument(
        "command", nargs="?", default="help", choices=["train", "evaluate", "run", "help"], help="Command to execute"
    )

    parser.add_argument("--config", type=str, help="Configuration file path")

    args = parser.parse_args()

    if args.command == "help" or args.command is None:
        parser.print_help()
        return 0

    print(f"Agent Forge CLI - Command: {args.command}")

    # For now, just print the command. In a full implementation,
    # this would import and call the appropriate modules from
    # the agent_forge package

    try:
        # Import the main agent forge functionality

        if args.command == "train":
            print("Starting training pipeline...")
            # run_pipeline("train", args.config)

        elif args.command == "evaluate":
            print("Starting evaluation pipeline...")
            # run_pipeline("evaluate", args.config)

        elif args.command == "run":
            print("Starting agent forge pipeline...")
            # run_pipeline("run", args.config)

    except ImportError as e:
        print(f"Warning: Could not import agent_forge modules: {e}")
        print("Agent Forge system may not be fully configured.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
