"""Core Digital Twin module with minimal implementation.

This simplified version provides the basic data structures and
APIs needed for import validation and lightweight tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LearningProfile:
    """Basic learner profile used by the digital twin."""

    student_id: str
    name: str
    age: int = 0
    grade_level: int = 0
    language: str = "en"
    region: str = ""
    learning_style: str = "visual"
    interests: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSession:
    """Record of a learning session."""

    session_id: str
    student_id: str
    tutor_model_id: str
    start_time: str = ""
    end_time: str = ""
    duration_minutes: int = 0
    concepts_covered: List[str] = field(default_factory=list)


@dataclass
class KnowledgeState:
    """Simple representation of student knowledge."""

    student_id: str
    subject: str
    concept: str
    mastery_level: float = 0.0


class ShadowSimulator:
    """Very small placeholder for path exploration logic."""

    def explore_paths(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return []


class DigitalTwin:
    """Light‑weight Digital Twin used for import validation."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.project_name = self.config.get("twin_id", "aivillage-digital-twin")
        self.students: Dict[str, LearningProfile] = {}
        self.shadow_simulator = ShadowSimulator()

    def update_profile(self, profile: LearningProfile) -> None:
        """Add or update a learner profile."""
        self.students[profile.student_id] = profile

    def get_preferences(self, student_id: str) -> Dict[str, Any]:
        """Return stored preferences for the given student."""
        profile = self.students.get(student_id)
        return profile.preferences if profile else {}
