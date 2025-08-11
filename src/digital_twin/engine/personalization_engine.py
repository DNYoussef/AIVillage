"""Minimal personalization engine."""

from __future__ import annotations

from typing import Any

from typing import Dict


class PersonalizationEngine:
    """Trivial content personalization placeholder."""

    def __init__(self) -> None:
        self.vectors: Dict[str, Any] = {}

    def personalize(self, user_id: str, content: Any) -> Any:
        """Return ``content`` unchanged; real implementation would adapt it."""
        self.vectors.setdefault(user_id, [])
        return content
