"""Simple in-memory profile manager for the Digital Twin."""

from __future__ import annotations

from typing import Dict, List, Optional

from .digital_twin import LearningProfile


class ProfileManager:
    """Manages ``LearningProfile`` instances."""

    def __init__(self) -> None:
        self._profiles: Dict[str, LearningProfile] = {}

    def add_profile(self, profile: LearningProfile) -> None:
        self._profiles[profile.student_id] = profile

    def get_profile(self, student_id: str) -> Optional[LearningProfile]:
        return self._profiles.get(student_id)

    def list_profiles(self) -> List[LearningProfile]:
        return list(self._profiles.values())
