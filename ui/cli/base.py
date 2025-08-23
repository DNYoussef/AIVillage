"""Shared CLI parser and dispatch helpers."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable

Action = Callable[[argparse.Namespace], int]
ActionMap = dict[str, Action]


def build_parser(
    description: str,
    commands: Iterable[str],
    configure: Callable[[argparse.ArgumentParser], None] | None = None,
    command_name: str = "action",
    command_help: str | None = None,
) -> argparse.ArgumentParser:
    """Create a basic :class:`argparse.ArgumentParser`.

    Args:
        description: CLI description for the parser.
        commands: Valid command names for the positional argument.
        configure: Optional callback to add additional arguments.
        command_name: Name of the primary positional argument.
        command_help: Help text for the primary positional argument.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        command_name,
        choices=list(commands),
        help=command_help or f"{command_name} to perform",
    )
    if configure:
        configure(parser)
    return parser


def dispatch(commands: ActionMap, args: argparse.Namespace, attr: str = "action") -> int:
    """Dispatch to the handler mapped by ``attr`` in ``commands``."""
    key = getattr(args, attr)
    handler = commands.get(key)
    if handler:
        return handler(args)
    print(f"Error: Unknown {attr} '{key}'")
    return 1


def run_cli(
    description: str,
    commands: ActionMap,
    configure: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
    command_name: str = "action",
    verbose_attr: str = "verbose",
) -> int:
    """Run a CLI using shared parser/dispatch logic."""
    parser = build_parser(description, commands.keys(), configure, command_name)
    args = parser.parse_args(argv)
    if verbose_attr and getattr(args, verbose_attr, False):
        print(f"{description}: {getattr(args, command_name)}")
    return dispatch(commands, args, attr=command_name)
