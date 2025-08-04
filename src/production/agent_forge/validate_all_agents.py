"""Validation utilities for agent templates.

This module provides a CLI that instantiates every agent defined in
``production/agent_forge/templates/agents`` and performs lightweight
verification of their behaviour. The checks are intentionally simple and
focus on ensuring that templates can be loaded, that agents can communicate
via the project's messaging protocol, that KPI tracking updates after task
execution and that each agent reports the correct specialization role.

Usage
-----
Run the full validation suite from the repository root with::

    python -m agent_forge.validate_all_agents --full-test
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from src.communications.message import Message, MessageType
from src.communications.protocol import StandardCommunicationProtocol

from .agent_factory import AgentFactory


async def _exercise_agent(agent: Any) -> dict[str, Any]:
    """Send a test message to ``agent`` using the standard protocol.

    The agent's ``process`` method is invoked when the message is received and
    its performance history is updated. The resulting output dictionary is
    returned for verification.
    """
    protocol = StandardCommunicationProtocol()
    result: dict[str, Any] = {}

    async def handler(msg: Message) -> None:
        nonlocal result
        result = agent.process(msg.content)
        agent.update_performance(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "metrics": {},
            }
        )

    protocol.subscribe(agent.name, handler)
    message = Message(
        type=MessageType.TASK,
        sender="tester",
        receiver=agent.name,
        content={"task": "ping"},
    )
    await protocol.send_message(message)
    return result


def validate_all_agents(full_test: bool = False) -> dict[str, dict[str, bool]]:
    """Validate all agent templates.

    Parameters
    ----------
    full_test:
        Currently unused but retained for CLI compatibility. When ``True`` the
        function performs the full validation suite.

    Returns:
    -------
    dict
        Mapping of agent id to a dictionary describing which checks passed.
    """
    template_path = Path(__file__).resolve().parent / "templates"
    factory = AgentFactory(template_dir=str(template_path))
    results: dict[str, dict[str, bool]] = {}

    for agent_id in factory.templates:
        status = {
            "created": False,
            "communication": False,
            "kpi": False,
            "specialization": False,
        }
        try:
            agent = factory.create_agent(agent_id)
            status["created"] = True

            role = agent.specialization.role
            status["specialization"] = getattr(role, "value", role) == agent_id

            output = asyncio.run(_exercise_agent(agent))
            status["communication"] = output.get("status") == "completed"
            status["kpi"] = bool(agent.kpi_scores)
        except Exception:  # noqa: BLE001
            pass

        results[agent_id] = status

    return results


def main() -> None:
    """CLI entry point for agent validation."""
    parser = argparse.ArgumentParser(description="Validate agent templates")
    parser.add_argument(
        "--full-test", action="store_true", help="Run the full validation suite"
    )
    args = parser.parse_args()

    validation_results = validate_all_agents(full_test=args.full_test)
    failures = {
        agent: checks
        for agent, checks in validation_results.items()
        if not all(checks.values())
    }

    if failures:
        for agent, checks in failures.items():
            print(f"Validation failed for {agent}: {checks}")
        raise SystemExit(1)

    print("All agents validated successfully")


if __name__ == "__main__":  # pragma: no cover
    main()
