"""
Knowledge Retrieval Interface

Defines the contract for knowledge retrieval systems,
abstracting the specific implementation details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryMode(Enum):
    """Query processing modes"""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


@dataclass
class RetrievalResult:
    """Standard result format for knowledge retrieval"""

    id: str
    content: str
    source: str
    relevance_score: float
    confidence_score: float
    metadata: dict[str, Any]


@dataclass
class QueryContext:
    """Context information for knowledge queries"""

    user_id: str | None = None
    session_id: str | None = None
    domain: str | None = None
    preferences: dict[str, Any] = None


class KnowledgeRetrievalInterface(ABC):
    """
    Abstract interface for knowledge retrieval systems

    Defines the contract that all knowledge retrieval implementations
    must satisfy, enabling clean separation of business logic from
    infrastructure concerns.
    """

    @abstractmethod
    async def query(
        self,
        query: str,
        mode: QueryMode = QueryMode.BALANCED,
        max_results: int = 10,
        context: QueryContext | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant knowledge for a query

        Args:
            query: Natural language query
            mode: Processing mode affecting quality vs speed
            max_results: Maximum number of results to return
            context: Additional context for retrieval

        Returns:
            List of relevant knowledge items
        """
        pass

    @abstractmethod
    async def store_knowledge(
        self, content: str, title: str, metadata: dict[str, Any], knowledge_type: str = "document"
    ) -> str:
        """
        Store new knowledge in the system

        Args:
            content: The knowledge content
            title: Knowledge title/identifier
            metadata: Additional metadata
            knowledge_type: Type of knowledge (document, fact, etc.)

        Returns:
            Unique identifier for stored knowledge
        """
        pass

    @abstractmethod
    async def update_knowledge(
        self, knowledge_id: str, content: str | None = None, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Update existing knowledge

        Args:
            knowledge_id: ID of knowledge to update
            content: New content (if updating)
            metadata: New metadata (if updating)

        Returns:
            True if update successful
        """
        pass

    @abstractmethod
    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the system

        Args:
            knowledge_id: ID of knowledge to delete

        Returns:
            True if deletion successful
        """
        pass

    @abstractmethod
    async def get_knowledge_by_id(self, knowledge_id: str) -> RetrievalResult | None:
        """
        Retrieve specific knowledge by ID

        Args:
            knowledge_id: ID of knowledge to retrieve

        Returns:
            Knowledge item if found, None otherwise
        """
        pass

    @abstractmethod
    async def search_similar(
        self, reference_content: str, similarity_threshold: float = 0.7, max_results: int = 10
    ) -> list[RetrievalResult]:
        """
        Find knowledge similar to reference content

        Args:
            reference_content: Content to find similar items for
            similarity_threshold: Minimum similarity score
            max_results: Maximum results to return

        Returns:
            List of similar knowledge items
        """
        pass

    @abstractmethod
    async def get_system_stats(self) -> dict[str, Any]:
        """
        Get retrieval system statistics

        Returns:
            Dictionary with system metrics and health info
        """
        pass
