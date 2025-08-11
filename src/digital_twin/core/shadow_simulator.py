"""Placeholder shadow simulator."""

from __future__ import annotations

from typing import Any, Dict


class ShadowSimulator:
    """Very small simulator that records simulation requests."""

    def __init__(self) -> None:
        self.simulations: Dict[str, Any] = {}

    def simulate(self, profile: Any, scenario: str) -> None:
        """Store the scenario keyed by the profile's identifier."""
        user_id = getattr(profile, "student_id", "unknown")
        self.simulations[user_id] = scenario
