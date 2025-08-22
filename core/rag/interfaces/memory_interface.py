"""
Memory Interface

Defines the contract for memory management systems that handle
storage, retrieval, and lifecycle management of contextual information.
Built upon the established KnowledgeRetrievalInterface patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class MemoryType(Enum):
    """Types of memory storage"""

    SHORT_TERM = "short_term"  # Session-based temporary memory
    WORKING = "working"  # Active processing memory
    LONG_TERM = "long_term"  # Persistent cross-session memory
    EPISODIC = "episodic"  # Event and experience memory
    SEMANTIC = "semantic"  # Factual knowledge memory


class AccessPattern(Enum):
    """Memory access patterns for optimization"""

    SEQUENTIAL = "sequential"  # Sequential access pattern
    RANDOM = "random"  # Random access pattern
    TEMPORAL = "temporal"  # Time-based access pattern
    ASSOCIATIVE = "associative"  # Association-based access
    HIERARCHICAL = "hierarchical"  # Tree-like access pattern


class RetentionPolicy(Enum):
    """Memory retention and cleanup policies"""

    TIME_BASED = "time_based"  # Expire after time period
    ACCESS_BASED = "access_based"  # Expire based on access frequency
    CAPACITY_BASED = "capacity_based"  # LRU when capacity exceeded
    IMPORTANCE_BASED = "importance_based"  # Retain based on importance
    MANUAL = "manual"  # Manual retention control


@dataclass
class MemoryContext:
    """Context for memory operations"""

    user_id: str | None = None
    session_id: str | None = None
    domain: str | None = None
    retention_policy: RetentionPolicy = RetentionPolicy.TIME_BASED
    access_pattern: AccessPattern = AccessPattern.RANDOM
    encryption_required: bool = False


@dataclass
class MemoryMetadata:
    """Metadata for memory entries"""

    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance_score: float
    tags: list[str]
    associations: list[str]
    encryption_key_id: str | None = None


@dataclass
class MemoryEntry:
    """Individual memory entry"""

    memory_id: str
    content: Any
    memory_type: MemoryType
    metadata: MemoryMetadata
    expires_at: datetime | None = None
    is_encrypted: bool = False


@dataclass
class MemorySearchResult:
    """Result of memory search operation"""

    entries: list[MemoryEntry]
    total_matches: int
    search_latency_ms: float
    associations_found: list[str]
    metadata: dict[str, Any]


class MemoryInterface(ABC):
    """
    Abstract interface for memory management systems

    Defines the contract for systems that manage contextual memory,
    including storage, retrieval, associations, and lifecycle management.
    Follows the established patterns from KnowledgeRetrievalInterface.
    """

    @abstractmethod
    async def store(
        self,
        content: Any,
        memory_type: MemoryType,
        context: MemoryContext,
        ttl: timedelta | None = None,
        importance: float = 0.5,
    ) -> str:
        """
        Store content in memory with specified characteristics

        Args:
            content: Data to store in memory
            memory_type: Type of memory storage
            context: Storage context and constraints
            ttl: Time-to-live for memory entry
            importance: Importance score (0.0-1.0)

        Returns:
            Unique memory identifier
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        memory_id: str,
        update_access: bool = True,
    ) -> MemoryEntry | None:
        """
        Retrieve specific memory entry by identifier

        Args:
            memory_id: Unique identifier of memory entry
            update_access: Whether to update access statistics

        Returns:
            Memory entry if found, None otherwise
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        context: MemoryContext | None = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
    ) -> MemorySearchResult:
        """
        Search memory using natural language query

        Args:
            query: Search query
            memory_types: Types of memory to search
            context: Search context constraints
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity score

        Returns:
            Search results with matching memory entries
        """
        pass

    @abstractmethod
    async def associate(
        self,
        memory_id1: str,
        memory_id2: str,
        association_strength: float = 1.0,
        association_type: str = "related",
    ) -> bool:
        """
        Create association between memory entries

        Args:
            memory_id1: First memory entry identifier
            memory_id2: Second memory entry identifier
            association_strength: Strength of association (0.0-1.0)
            association_type: Type of association

        Returns:
            True if association created successfully
        """
        pass

    @abstractmethod
    async def get_associations(
        self,
        memory_id: str,
        association_types: list[str] | None = None,
        max_associations: int = 10,
    ) -> list[MemoryEntry]:
        """
        Get associated memory entries

        Args:
            memory_id: Memory entry to find associations for
            association_types: Types of associations to include
            max_associations: Maximum associations to return

        Returns:
            List of associated memory entries
        """
        pass

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: Any | None = None,
        importance: float | None = None,
        ttl: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """
        Update existing memory entry

        Args:
            memory_id: Memory entry to update
            content: New content (if updating)
            importance: New importance score (if updating)
            ttl: New time-to-live (if updating)
            tags: New tags (if updating)

        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        cascade_associations: bool = False,
    ) -> bool:
        """
        Delete memory entry from system

        Args:
            memory_id: Memory entry to delete
            cascade_associations: Whether to remove associations

        Returns:
            True if deletion successful
        """
        pass

    @abstractmethod
    async def cleanup_expired(
        self,
        memory_types: list[MemoryType] | None = None,
        force_cleanup: bool = False,
    ) -> dict[str, int]:
        """
        Clean up expired memory entries

        Args:
            memory_types: Types of memory to clean up
            force_cleanup: Force cleanup regardless of policy

        Returns:
            Dictionary with cleanup statistics
        """
        pass

    @abstractmethod
    async def get_memory_usage(
        self,
        context: MemoryContext | None = None,
    ) -> dict[str, Any]:
        """
        Get memory usage statistics and health metrics

        Args:
            context: Context to get statistics for

        Returns:
            Dictionary with memory usage and performance metrics
        """
        pass

    @abstractmethod
    async def optimize_storage(
        self,
        memory_types: list[MemoryType] | None = None,
        optimization_strategy: str = "balanced",
    ) -> dict[str, Any]:
        """
        Optimize memory storage and access patterns

        Args:
            memory_types: Types of memory to optimize
            optimization_strategy: Strategy for optimization

        Returns:
            Dictionary with optimization results and metrics
        """
        pass
