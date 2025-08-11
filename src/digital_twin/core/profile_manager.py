"""Minimal profile manager for the Digital Twin."""

from __future__ import annotations

from typing import Any, Dict


class ProfileManager:
    """Create and retrieve simple user profiles."""

    def __init__(self) -> None:
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def create_profile(self, user_id: str) -> None:
        """Create an empty profile for ``user_id``."""
        self.profiles[user_id] = {}

    def get_profile(self, user_id: str) -> Dict[str, Any] | None:
        """Return the stored profile if it exists."""
        return self.profiles.get(user_id)
