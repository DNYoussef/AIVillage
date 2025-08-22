"""Quiet-STaR reflection system for agent introspection.

Handles personal journal and reflection capabilities following single
responsibility principle. Extracted from BaseAgentTemplate to reduce
coupling and improve maintainability.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..agent_constants import ReflectionConstants, ReflectionType

logger = logging.getLogger(__name__)


@dataclass
class QuietStarReflection:
    """Quiet-star reflection entry for personal journal."""

    reflection_id: str
    timestamp: datetime
    reflection_type: ReflectionType
    context: str  # What was happening
    thoughts: str  # <|startofthought|> content <|endofthought|>
    insights: str  # Key learnings
    emotional_valence: float  # -1.0 to 1.0
    unexpectedness_score: float  # 0.0 to 1.0 (for Langroid memory)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert reflection to dictionary for serialization."""
        return {
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp.isoformat(),
            "reflection_type": self.reflection_type.value,
            "context": self.context,
            "thoughts": self.thoughts,
            "insights": self.insights,
            "emotional_valence": self.emotional_valence,
            "unexpectedness_score": self.unexpectedness_score,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuietStarReflection":
        """Create reflection from dictionary."""
        return cls(
            reflection_id=data["reflection_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reflection_type=ReflectionType(data["reflection_type"]),
            context=data["context"],
            thoughts=data["thoughts"],
            insights=data["insights"],
            emotional_valence=data["emotional_valence"],
            unexpectedness_score=data["unexpectedness_score"],
            tags=data.get("tags", []),
        )


class ReflectionManager:
    """Manages quiet-star reflection and personal journaling.

    Single responsibility: Handle reflection creation, storage, and retrieval.
    Uses dependency injection for storage and memory integration.
    """

    def __init__(self, max_entries: int | None = None):
        """Initialize reflection manager.

        Args:
            max_entries: Maximum number of reflections to keep in memory
        """
        self._personal_journal: list[QuietStarReflection] = []
        self._reflection_count = 0
        self._max_entries = max_entries or ReflectionConstants.MEMORY_STORAGE_THRESHOLD

    async def record_reflection(
        self,
        reflection_type: ReflectionType,
        context: str,
        raw_thoughts: str,
        insights: str,
        emotional_valence: float = 0.0,
        tags: list[str] | None = None,
    ) -> str:
        """Record a quiet-star reflection in personal journal.

        Args:
            reflection_type: Type of reflection
            context: What was happening
            raw_thoughts: Raw thought process
            insights: Key insights gained
            emotional_valence: Emotional rating (-1.0 to 1.0)
            tags: Optional tags for categorization

        Returns:
            Reflection ID
        """
        # Calculate unexpectedness for memory system integration
        unexpectedness_score = self._calculate_unexpectedness(context, insights, emotional_valence)

        reflection = QuietStarReflection(
            reflection_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            reflection_type=reflection_type,
            context=context,
            thoughts=f"<|startofthought|>{raw_thoughts}<|endofthought|>",
            insights=insights,
            emotional_valence=emotional_valence,
            unexpectedness_score=unexpectedness_score,
            tags=tags or [],
        )

        self._personal_journal.append(reflection)
        self._reflection_count += 1

        # Manage memory by keeping only recent entries
        if len(self._personal_journal) > self._max_entries:
            self._personal_journal = self._personal_journal[-self._max_entries :]

        logger.info(f"Quiet-STaR reflection recorded: {reflection.reflection_id}")
        return reflection.reflection_id

    def _calculate_unexpectedness(self, context: str, insights: str, emotional_valence: float) -> float:
        """Calculate unexpectedness score for memory system integration.

        Uses configurable weights and thresholds to avoid magic numbers.

        Args:
            context: Context of the reflection
            insights: Insights gained
            emotional_valence: Emotional valence

        Returns:
            Unexpectedness score (0.0 to 1.0)
        """
        base_score = 0.0

        # High absolute emotional valence indicates unexpectedness
        base_score += abs(emotional_valence) * ReflectionConstants.EMOTIONAL_VALENCE_WEIGHT

        # Look for surprise indicators in context and insights
        text_to_check = f"{context} {insights}".lower()

        for word in ReflectionConstants.SURPRISE_WORDS:
            if word in text_to_check:
                base_score += ReflectionConstants.SURPRISE_WORD_WEIGHT

        # Length and detail often indicate significance
        if len(insights) > ReflectionConstants.DETAILED_INSIGHT_MIN_LENGTH:
            base_score += ReflectionConstants.DETAILED_INSIGHT_WEIGHT

        return min(1.0, base_score)  # Cap at 1.0

    def get_recent_reflections(
        self, count: int = 10, reflection_type: ReflectionType | None = None
    ) -> list[QuietStarReflection]:
        """Get recent reflections, optionally filtered by type.

        Args:
            count: Number of reflections to return
            reflection_type: Optional filter by reflection type

        Returns:
            List of recent reflections
        """
        reflections = self._personal_journal

        if reflection_type:
            reflections = [r for r in reflections if r.reflection_type == reflection_type]

        return reflections[-count:] if reflections else []

    def get_reflection_by_id(self, reflection_id: str) -> QuietStarReflection | None:
        """Get reflection by ID.

        Args:
            reflection_id: Reflection ID to find

        Returns:
            Reflection if found, None otherwise
        """
        for reflection in self._personal_journal:
            if reflection.reflection_id == reflection_id:
                return reflection
        return None

    def search_reflections(self, query: str, max_results: int = 5) -> list[QuietStarReflection]:
        """Search reflections by text content.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of matching reflections
        """
        query_lower = query.lower()
        matches = []

        for reflection in self._personal_journal:
            # Search in context and insights
            searchable_text = f"{reflection.context} {reflection.insights}".lower()
            if query_lower in searchable_text:
                matches.append(reflection)

        # Return most recent matches first
        return matches[-max_results:] if matches else []

    def get_reflection_stats(self) -> dict[str, Any]:
        """Get reflection statistics.

        Returns:
            Dictionary with reflection statistics
        """
        if not self._personal_journal:
            return {
                "total_reflections": 0,
                "reflection_types": {},
                "avg_emotional_valence": 0.0,
                "avg_unexpectedness": 0.0,
            }

        # Count by type
        type_counts = {}
        for reflection in self._personal_journal:
            type_name = reflection.reflection_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Calculate averages
        total_valence = sum(r.emotional_valence for r in self._personal_journal)
        total_unexpectedness = sum(r.unexpectedness_score for r in self._personal_journal)
        count = len(self._personal_journal)

        return {
            "total_reflections": count,
            "reflection_types": type_counts,
            "avg_emotional_valence": total_valence / count,
            "avg_unexpectedness": total_unexpectedness / count,
        }

    def export_reflections(self) -> list[dict[str, Any]]:
        """Export all reflections as dictionaries.

        Returns:
            List of reflection dictionaries
        """
        return [reflection.to_dict() for reflection in self._personal_journal]

    def import_reflections(self, reflection_data: list[dict[str, Any]]) -> int:
        """Import reflections from dictionaries.

        Args:
            reflection_data: List of reflection dictionaries

        Returns:
            Number of reflections imported
        """
        imported_count = 0

        for data in reflection_data:
            try:
                reflection = QuietStarReflection.from_dict(data)
                self._personal_journal.append(reflection)
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import reflection: {e}")
                continue

        # Update counter
        self._reflection_count = len(self._personal_journal)

        # Maintain size limit
        if len(self._personal_journal) > self._max_entries:
            self._personal_journal = self._personal_journal[-self._max_entries :]

        logger.info(f"Imported {imported_count} reflections")
        return imported_count

    @property
    def reflection_count(self) -> int:
        """Get total reflection count."""
        return self._reflection_count

    @property
    def journal_size(self) -> int:
        """Get current journal size."""
        return len(self._personal_journal)
