"""Langroid-based personal memory system for agents.

Implements emotional memory based on unexpectedness following the
single responsibility principle. Extracted from BaseAgentTemplate
to improve maintainability and enable testing.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any
import uuid

from ..agent_constants import MemoryConstants, MemoryImportance

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Langroid-based memory entry focused on unexpectedness."""

    memory_id: str
    timestamp: datetime
    content: str
    importance: MemoryImportance
    unexpectedness_score: float  # Core metric for Langroid system
    emotional_context: dict[str, float]  # Multiple emotional dimensions
    associated_agents: list[str]
    retrieval_count: int = 0
    last_accessed: datetime | None = None

    def decay_importance(self, time_elapsed_hours: float) -> float:
        """Calculate decayed importance based on time and retrieval.

        Uses configurable parameters to avoid magic numbers.

        Args:
            time_elapsed_hours: Hours since memory creation

        Returns:
            Decayed importance score
        """
        # Weekly decay with minimum floor
        base_decay = max(
            MemoryConstants.MIN_DECAY_FACTOR, 1.0 - (time_elapsed_hours / MemoryConstants.WEEKLY_DECAY_HOURS)
        )

        # Retrieval boost (memories accessed more get stronger)
        retrieval_boost = min(
            MemoryConstants.MAX_RETRIEVAL_BOOST, 1.0 + (self.retrieval_count * MemoryConstants.RETRIEVAL_BOOST_FACTOR)
        )

        return self.importance.value * base_decay * retrieval_boost

    def to_dict(self) -> dict[str, Any]:
        """Convert memory entry to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "importance": self.importance.value,
            "unexpectedness_score": self.unexpectedness_score,
            "emotional_context": self.emotional_context,
            "associated_agents": self.associated_agents,
            "retrieval_count": self.retrieval_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            importance=MemoryImportance(data["importance"]),
            unexpectedness_score=data["unexpectedness_score"],
            emotional_context=data["emotional_context"],
            associated_agents=data["associated_agents"],
            retrieval_count=data.get("retrieval_count", 0),
            last_accessed=(datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None),
        )


class LangroidMemoryManager:
    """Manages Langroid-based emotional memory system.

    Single responsibility: Handle memory storage, retrieval, and decay
    based on unexpectedness scores and emotional context.
    """

    def __init__(self, max_entries: int | None = None, retrieval_threshold: float | None = None):
        """Initialize memory manager.

        Args:
            max_entries: Maximum memory entries to keep
            retrieval_threshold: Minimum importance for retrieval
        """
        self._personal_memory: list[MemoryEntry] = []
        self._max_entries = max_entries or MemoryConstants.MAX_MEMORY_ENTRIES
        self._retrieval_threshold = retrieval_threshold or MemoryConstants.DEFAULT_RETRIEVAL_THRESHOLD

    def store_memory(
        self,
        content: str,
        unexpectedness_score: float,
        emotional_context: dict[str, float],
        associated_agents: list[str] | None = None,
    ) -> str:
        """Store a new memory entry.

        Args:
            content: Memory content
            unexpectedness_score: Unexpectedness score (0.0-1.0)
            emotional_context: Emotional dimensions
            associated_agents: List of associated agent IDs

        Returns:
            Memory ID
        """
        # Determine importance based on unexpectedness
        importance = self._calculate_importance(unexpectedness_score)

        memory_entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            content=content,
            importance=importance,
            unexpectedness_score=unexpectedness_score,
            emotional_context=emotional_context,
            associated_agents=associated_agents or [],
        )

        self._personal_memory.append(memory_entry)

        # Manage memory size
        if len(self._personal_memory) > self._max_entries:
            # Remove oldest, least important memories
            self._personal_memory.sort(key=lambda m: (m.importance.value, m.timestamp))
            self._personal_memory = self._personal_memory[-self._max_entries :]

        logger.info(f"Stored memory: {memory_entry.memory_id}, " f"importance: {importance.name}")
        return memory_entry.memory_id

    def _calculate_importance(self, unexpectedness_score: float) -> MemoryImportance:
        """Calculate memory importance from unexpectedness score.

        Uses configurable thresholds to avoid magic numbers.

        Args:
            unexpectedness_score: Unexpectedness score (0.0-1.0)

        Returns:
            Memory importance level
        """
        from ..agent_constants import ReflectionConstants

        if unexpectedness_score >= ReflectionConstants.TRANSFORMATIVE_THRESHOLD:
            return MemoryImportance.TRANSFORMATIVE
        elif unexpectedness_score >= ReflectionConstants.CRITICAL_THRESHOLD:
            return MemoryImportance.CRITICAL
        elif unexpectedness_score >= ReflectionConstants.IMPORTANT_THRESHOLD:
            return MemoryImportance.IMPORTANT
        elif unexpectedness_score >= ReflectionConstants.NOTABLE_THRESHOLD:
            return MemoryImportance.NOTABLE
        else:
            return MemoryImportance.ROUTINE

    def retrieve_similar_memories(
        self, query: str, importance_threshold: float | None = None, max_memories: int = 5
    ) -> list[MemoryEntry]:
        """Retrieve similar memories based on content similarity.

        Args:
            query: Query string for similarity matching
            importance_threshold: Minimum importance threshold
            max_memories: Maximum number of memories to return

        Returns:
            List of relevant memory entries
        """
        threshold = importance_threshold or self._retrieval_threshold
        current_time = datetime.now()

        # Filter and score memories
        relevant_memories: list[tuple[MemoryEntry, float]] = []

        for memory in self._personal_memory:
            # Calculate decayed importance
            time_elapsed = (current_time - memory.timestamp).total_seconds() / 3600
            decayed_importance = memory.decay_importance(time_elapsed)

            if decayed_importance >= threshold:
                # Calculate similarity
                similarity = self._calculate_memory_similarity(query, memory.content)
                if similarity > MemoryConstants.MIN_SIMILARITY_THRESHOLD:
                    # Update access tracking
                    memory.last_accessed = current_time
                    memory.retrieval_count += 1
                    relevant_memories.append((memory, similarity))

        # Sort by similarity and return top matches
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:max_memories]]

    def _calculate_memory_similarity(self, query: str, memory_content: str) -> float:
        """Calculate similarity between query and memory content.

        Simple implementation - in production would use embeddings.

        Args:
            query: Query string
            memory_content: Memory content to compare

        Returns:
            Similarity score (0.0-1.0)
        """
        query_words = set(query.lower().split())
        memory_words = set(memory_content.lower().split())

        if not query_words or not memory_words:
            return 0.0

        intersection = query_words.intersection(memory_words)
        union = query_words.union(memory_words)

        return len(intersection) / len(union) if union else 0.0

    def get_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Get memory entry by ID.

        Args:
            memory_id: Memory ID to find

        Returns:
            Memory entry if found, None otherwise
        """
        for memory in self._personal_memory:
            if memory.memory_id == memory_id:
                return memory
        return None

    def get_memories_by_importance(self, min_importance: MemoryImportance, max_results: int = 10) -> list[MemoryEntry]:
        """Get memories above a certain importance level.

        Args:
            min_importance: Minimum importance level
            max_results: Maximum results to return

        Returns:
            List of important memories
        """
        important_memories = [
            memory for memory in self._personal_memory if memory.importance.value >= min_importance.value
        ]

        # Sort by importance and recency
        important_memories.sort(key=lambda m: (m.importance.value, m.timestamp), reverse=True)

        return important_memories[:max_results]

    def cleanup_old_memories(self, max_age_days: int = 365) -> int:
        """Remove very old, low-importance memories.

        Args:
            max_age_days: Maximum age for routine memories

        Returns:
            Number of memories removed
        """
        current_time = datetime.now()
        initial_count = len(self._personal_memory)

        # Keep memories that are either important or recent
        self._personal_memory = [
            memory
            for memory in self._personal_memory
            if (
                memory.importance.value >= MemoryImportance.IMPORTANT.value
                or (current_time - memory.timestamp).days <= max_age_days
            )
        ]

        removed_count = initial_count - len(self._personal_memory)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old memories")

        return removed_count

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self._personal_memory:
            return {
                "total_memories": 0,
                "importance_distribution": {},
                "avg_unexpectedness": 0.0,
                "most_accessed": None,
            }

        # Count by importance
        importance_counts = {}
        for memory in self._personal_memory:
            importance_name = memory.importance.name
            importance_counts[importance_name] = importance_counts.get(importance_name, 0) + 1

        # Calculate averages
        total_unexpectedness = sum(m.unexpectedness_score for m in self._personal_memory)
        avg_unexpectedness = total_unexpectedness / len(self._personal_memory)

        # Find most accessed memory
        most_accessed = max(self._personal_memory, key=lambda m: m.retrieval_count, default=None)

        return {
            "total_memories": len(self._personal_memory),
            "importance_distribution": importance_counts,
            "avg_unexpectedness": avg_unexpectedness,
            "most_accessed": (
                {
                    "memory_id": most_accessed.memory_id,
                    "retrieval_count": most_accessed.retrieval_count,
                    "content_preview": (
                        most_accessed.content[:100] + "..."
                        if len(most_accessed.content) > 100
                        else most_accessed.content
                    ),
                }
                if most_accessed
                else None
            ),
        }

    def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories as dictionaries.

        Returns:
            List of memory dictionaries
        """
        return [memory.to_dict() for memory in self._personal_memory]

    def import_memories(self, memory_data: list[dict[str, Any]]) -> int:
        """Import memories from dictionaries.

        Args:
            memory_data: List of memory dictionaries

        Returns:
            Number of memories imported
        """
        imported_count = 0

        for data in memory_data:
            try:
                memory = MemoryEntry.from_dict(data)
                self._personal_memory.append(memory)
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")
                continue

        # Maintain size limit
        if len(self._personal_memory) > self._max_entries:
            self._personal_memory.sort(key=lambda m: (m.importance.value, m.timestamp))
            self._personal_memory = self._personal_memory[-self._max_entries :]

        logger.info(f"Imported {imported_count} memories")
        return imported_count

    @property
    def memory_count(self) -> int:
        """Get current memory count."""
        return len(self._personal_memory)

    @property
    def retrieval_threshold(self) -> float:
        """Get current retrieval threshold."""
        return self._retrieval_threshold

    @retrieval_threshold.setter
    def retrieval_threshold(self, value: float) -> None:
        """Set retrieval threshold."""
        self._retrieval_threshold = max(0.0, min(1.0, value))
