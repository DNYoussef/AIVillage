"""Minimal personalization engine."""

from __future__ import annotations

from typing import Any

from ..core.digital_twin import LearningProfile


class PersonalizationEngine:
    """Trivial content personalization placeholder."""

    def personalize(self, profile: LearningProfile, content: Any) -> Any:
        """Return ``content`` unchanged; real implementation would adapt it."""
        return content
