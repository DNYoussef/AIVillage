"""
Knowledge Service

Provides knowledge management and retrieval capabilities.
This is a reference implementation to resolve import issues after reorganization.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class KnowledgeItem:
    """Knowledge item data structure."""

    knowledge_id: str
    content: str
    category: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeService:
    """
    Service for managing agent knowledge and information.

    This is a reference implementation to resolve import dependencies
    during the reorganization process.
    """

    def __init__(self):
        """Initialize the knowledge service."""
        self._knowledge_base: dict[str, KnowledgeItem] = {}

    def store_knowledge(self, knowledge_id: str, content: str, category: str) -> KnowledgeItem:
        """Store a knowledge item."""
        item = KnowledgeItem(knowledge_id=knowledge_id, content=content, category=category)
        self._knowledge_base[knowledge_id] = item
        return item

    def retrieve_knowledge(self, knowledge_id: str) -> KnowledgeItem | None:
        """Retrieve a knowledge item by ID."""
        return self._knowledge_base.get(knowledge_id)

    def search_knowledge(self, query: str, category: str | None = None) -> list[KnowledgeItem]:
        """Search knowledge items."""
        results = []
        for item in self._knowledge_base.values():
            if query.lower() in item.content.lower():
                if category is None or item.category == category:
                    results.append(item)
        return results

    def get_categories(self) -> list[str]:
        """Get all knowledge categories."""
        return list(set(item.category for item in self._knowledge_base.values()))
