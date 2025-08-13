"""Base classes and types for HypeRAG dual-memory system."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class MemoryType(Enum):
    """Types of memory storage."""

    EPISODIC = "episodic"  # Short-term, recent events
    SEMANTIC = "semantic"  # Long-term, consolidated knowledge
    WORKING = "working"  # Active processing


class ConfidenceType(Enum):
    """Types of confidence measurements."""

    BAYESIAN = "bayesian"  # Prior/posterior updates
    FREQUENCY = "frequency"  # Usage-based confidence
    TEMPORAL = "temporal"  # Time-decay confidence
    SOCIAL = "social"  # Community consensus


@dataclass
class Document:
    """Base document class for memory storage."""

    id: str
    content: str
    doc_type: str
    created_at: datetime
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    confidence: float = 1.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()


@dataclass
class Node:
    """Enhanced node with memory-specific properties."""

    id: str
    content: str
    node_type: str
    memory_type: MemoryType
    confidence: float = 1.0
    embedding: np.ndarray | None = None
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0
    user_id: str | None = None

    # GDC support
    gdc_flags: list[str] = field(default_factory=list)
    popularity_rank: int = 0

    # Temporal properties
    importance_score: float = 0.5
    decay_rate: float = 0.1
    ttl: int | None = None  # seconds

    # Uncertainty tracking
    uncertainty: float = 0.0
    confidence_type: ConfidenceType = ConfidenceType.BAYESIAN

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()

    def update_access(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def calculate_recency_weight(self, current_time: datetime | None = None) -> float:
        """Calculate time-based recency weight."""
        if not current_time:
            current_time = datetime.now()

        if not self.created_at:
            return 0.0

        time_diff = (current_time - self.created_at).total_seconds()
        return np.exp(-self.decay_rate * time_diff / 3600)  # Hourly decay

    def is_expired(self, current_time: datetime | None = None) -> bool:
        """Check if node has expired based on TTL."""
        if not self.ttl or not self.created_at:
            return False

        if not current_time:
            current_time = datetime.now()

        age_seconds = (current_time - self.created_at).total_seconds()
        return age_seconds > self.ttl


@dataclass
class Edge:
    """Enhanced edge with hypergraph and memory properties."""

    id: str
    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0

    # Hypergraph support (n-ary relationships)
    participants: list[str] = field(default_factory=list)  # All connected nodes

    # Memory properties
    memory_type: MemoryType = MemoryType.EPISODIC
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0

    # GDC and popularity
    gdc_flags: list[str] = field(default_factory=list)
    popularity_rank: int = 0

    # Personalization (Rel-GAT)
    alpha_weight: float | None = None
    user_id: str | None = None

    # Uncertainty and evidence
    uncertainty: float = 0.0
    evidence_count: int = 1
    source_docs: list[str] = field(default_factory=list)

    # Tags and metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()
        # Ensure participants includes source and target
        if self.source_id not in self.participants:
            self.participants.append(self.source_id)
        if self.target_id not in self.participants:
            self.participants.append(self.target_id)

    def update_access(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def add_evidence(self, doc_id: str, confidence_boost: float = 0.1) -> None:
        """Add evidence supporting this edge."""
        if doc_id not in self.source_docs:
            self.source_docs.append(doc_id)
            self.evidence_count += 1
            # Boost confidence with diminishing returns
            confidence_gain = confidence_boost * (1.0 / np.sqrt(self.evidence_count))
            self.confidence = min(1.0, self.confidence + confidence_gain)

    def update_bayesian_confidence(self, prior: float, likelihood: float) -> None:
        """Update confidence using Bayesian inference."""
        # Simple Bayesian update: posterior ∝ likelihood × prior
        posterior = (likelihood * prior) / (
            (likelihood * prior) + ((1 - likelihood) * (1 - prior))
        )
        self.confidence = posterior

    def is_hyperedge(self) -> bool:
        """Check if this is a hyperedge (connects >2 nodes)."""
        return len(self.participants) > 2


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend."""

    @abstractmethod
    async def close(self) -> None:
        """Close backend connections."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""


@dataclass
class QueryResult:
    """Result of a memory query."""

    nodes: list[Node]
    edges: list[Edge]
    total_count: int
    query_time_ms: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationBatch:
    """Batch of items for consolidation."""

    id: str
    nodes: list[Node]
    edges: list[Edge]
    confidence_threshold: float
    created_at: datetime
    status: str = "pending"  # pending, processing, completed, failed

    def __post_init__(self):
        if not self.id:
            self.id = f"batch_{int(datetime.now().timestamp())}"
        if not self.created_at:
            self.created_at = datetime.now()


@dataclass
class MemoryStats:
    """Statistics about memory usage."""

    total_nodes: int
    total_edges: int
    episodic_nodes: int
    semantic_nodes: int
    avg_confidence: float
    memory_usage_mb: float
    last_consolidation: datetime | None
    pending_consolidations: int


class EmbeddingManager:
    """Manages embeddings for memory systems."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text (mock implementation)."""
        # In production, this would use a real embedding model
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(self.dimension).astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        if a is None or b is None:
            return 0.0
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def find_similar(
        self, query_embedding: np.ndarray, candidates: list[np.ndarray], top_k: int = 10
    ) -> list[tuple]:
        """Find most similar embeddings."""
        if not candidates:
            return []

        similarities = []
        for i, candidate in enumerate(candidates):
            sim = self.cosine_similarity(query_embedding, candidate)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_episodic_node(
    content: str, user_id: str | None = None, ttl_hours: int = 168
) -> Node:  # 7 days default
    """Create a new episodic memory node."""
    return Node(
        id=str(uuid.uuid4()),
        content=content,
        node_type="episodic",
        memory_type=MemoryType.EPISODIC,
        user_id=user_id,
        ttl=ttl_hours * 3600,  # Convert to seconds
        importance_score=0.3,  # Lower for episodic
        decay_rate=0.2,  # Faster decay
        confidence_type=ConfidenceType.TEMPORAL,
    )


def create_semantic_node(content: str, confidence: float = 0.8) -> Node:
    """Create a new semantic memory node."""
    return Node(
        id=str(uuid.uuid4()),
        content=content,
        node_type="semantic",
        memory_type=MemoryType.SEMANTIC,
        confidence=confidence,
        importance_score=0.8,  # Higher for semantic
        decay_rate=0.01,  # Slower decay
        confidence_type=ConfidenceType.BAYESIAN,
    )


def create_hyperedge(
    participants: list[str],
    relation: str,
    confidence: float = 1.0,
    user_id: str | None = None,
) -> Edge:
    """Create a hyperedge connecting multiple nodes."""
    # Use first two participants as source/target for compatibility
    source_id = participants[0] if participants else str(uuid.uuid4())
    target_id = participants[1] if len(participants) > 1 else str(uuid.uuid4())

    return Edge(
        id=str(uuid.uuid4()),
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        confidence=confidence,
        participants=participants.copy(),
        user_id=user_id,
        evidence_count=1,
    )
