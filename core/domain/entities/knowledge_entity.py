"""
Knowledge Domain Entity

Represents knowledge artifacts, documents, and information
that agents can access and reason about.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class KnowledgeType(Enum):
    """Types of knowledge in the system"""

    DOCUMENT = "document"
    FACT = "fact"
    RULE = "rule"
    PROCEDURE = "procedure"
    PATTERN = "pattern"
    INSIGHT = "insight"
    MEMORY = "memory"


@dataclass
class KnowledgeId:
    """Knowledge identifier value object"""

    value: str

    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("KnowledgeId must be a non-empty string")

    @classmethod
    def generate(cls) -> KnowledgeId:
        """Generate new unique knowledge ID"""
        return cls(str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class Knowledge:
    """
    Core Knowledge domain entity

    Represents information and knowledge that can be stored,
    retrieved, and reasoned about by agents.
    """

    # Identity
    id: KnowledgeId
    title: str
    content: str
    knowledge_type: KnowledgeType

    # Classification and relationships
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_knowledge_ids: list[KnowledgeId] = field(default_factory=list)

    # Quality and trust metrics
    confidence_score: float = 1.0  # 0.0 to 1.0
    trust_score: float = 1.0  # 0.0 to 1.0
    relevance_score: float = 1.0  # 0.0 to 1.0

    # Provenance
    source: str | None = None
    author: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Usage tracking
    access_count: int = 0
    last_accessed: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate knowledge invariants"""
        if not self.title.strip():
            raise ValueError("Knowledge title cannot be empty")

        if not self.content.strip():
            raise ValueError("Knowledge content cannot be empty")

        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")

        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError("Trust score must be between 0.0 and 1.0")

        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")

    def add_topic(self, topic: str) -> None:
        """Add topic classification"""
        topic = topic.strip().lower()
        if topic and topic not in self.topics:
            self.topics.append(topic)

    def remove_topic(self, topic: str) -> None:
        """Remove topic classification"""
        topic = topic.strip().lower()
        if topic in self.topics:
            self.topics.remove(topic)

    def add_tag(self, tag: str) -> None:
        """Add tag to knowledge"""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from knowledge"""
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)

    def link_to_knowledge(self, knowledge_id: KnowledgeId) -> None:
        """Create relationship to other knowledge"""
        if knowledge_id != self.id and knowledge_id not in self.related_knowledge_ids:
            self.related_knowledge_ids.append(knowledge_id)

    def unlink_from_knowledge(self, knowledge_id: KnowledgeId) -> None:
        """Remove relationship to other knowledge"""
        if knowledge_id in self.related_knowledge_ids:
            self.related_knowledge_ids.remove(knowledge_id)

    def access(self) -> None:
        """Record access to this knowledge"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def update_content(self, new_content: str) -> None:
        """Update knowledge content"""
        if not new_content.strip():
            raise ValueError("New content cannot be empty")

        self.content = new_content
        self.last_updated = datetime.now()

    def update_quality_scores(
        self, confidence: float | None = None, trust: float | None = None, relevance: float | None = None
    ) -> None:
        """Update quality and trust metrics"""
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence score must be between 0.0 and 1.0")
            self.confidence_score = confidence

        if trust is not None:
            if not 0.0 <= trust <= 1.0:
                raise ValueError("Trust score must be between 0.0 and 1.0")
            self.trust_score = trust

        if relevance is not None:
            if not 0.0 <= relevance <= 1.0:
                raise ValueError("Relevance score must be between 0.0 and 1.0")
            self.relevance_score = relevance

        self.last_updated = datetime.now()

    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score"""
        return (self.confidence_score + self.trust_score + self.relevance_score) / 3.0

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if knowledge meets quality threshold"""
        return self.get_overall_quality_score() >= threshold

    def has_topic(self, topic: str) -> bool:
        """Check if knowledge contains specific topic"""
        return topic.strip().lower() in self.topics

    def has_tag(self, tag: str) -> bool:
        """Check if knowledge has specific tag"""
        return tag.strip().lower() in self.tags

    def to_dict(self) -> dict[str, Any]:
        """Convert knowledge to dictionary representation"""
        return {
            "id": str(self.id),
            "title": self.title,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "topics": self.topics,
            "tags": self.tags,
            "related_knowledge_ids": [str(kid) for kid in self.related_knowledge_ids],
            "confidence_score": self.confidence_score,
            "trust_score": self.trust_score,
            "relevance_score": self.relevance_score,
            "source": self.source,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Knowledge:
        """Create knowledge from dictionary representation"""
        return cls(
            id=KnowledgeId(data["id"]),
            title=data["title"],
            content=data["content"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            topics=data.get("topics", []),
            tags=data.get("tags", []),
            related_knowledge_ids=[KnowledgeId(kid) for kid in data.get("related_knowledge_ids", [])],
            confidence_score=data.get("confidence_score", 1.0),
            trust_score=data.get("trust_score", 1.0),
            relevance_score=data.get("relevance_score", 1.0),
            source=data.get("source"),
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            metadata=data.get("metadata", {}),
        )
