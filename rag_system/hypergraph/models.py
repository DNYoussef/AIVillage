"""
Hypergraph Models

Core data structures for the dual-memory hypergraph knowledge system.
Implements Hyperedge for n-ary relationships and HippoNode for episodic memory.
"""

from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, validator
import numpy as np
import uuid


class Hyperedge(BaseModel):
    """
    N-ary hyperedge representing complex relationships in knowledge graph.

    Unlike traditional graph edges (binary), hyperedges can connect 2+ entities,
    enabling representation of complex scenarios like:
    - Patient + Medication + Allergen + Contraindication
    - Query + Document + Answer + Confidence
    - Event + Participants + Location + Time
    """

    id: str = Field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:8]}")
    entities: List[str] = Field(min_items=2, description="List of entity IDs (minimum 2)")
    relation: str = Field(description="Relationship type connecting the entities")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_docs: List[str] = Field(default_factory=list, description="Source document IDs")
    embedding: Optional[np.ndarray] = Field(default=None, description="Vector embedding")

    # Additional metadata fields for different domains
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow numpy arrays in Pydantic model
        arbitrary_types_allowed = True
        # Use enum values in schema
        use_enum_values = True
        # Generate example for docs
        schema_extra = {
            "example": {
                "entities": ["patient_123", "medication_456", "ingredient_789"],
                "relation": "prescribed_containing_allergen",
                "confidence": 0.95,
                "source_docs": ["medical_record_001", "drug_database"]
            }
        }

    @validator('entities')
    def validate_entities(cls, v):
        """Ensure minimum entity count and no duplicates"""
        if len(v) < 2:
            raise ValueError("Hyperedge must connect at least 2 entities")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate entities not allowed in hyperedge")
        return v

    @validator('relation')
    def validate_relation(cls, v):
        """Ensure relation is non-empty and valid"""
        if not v or not v.strip():
            raise ValueError("Relation cannot be empty")
        return v.strip()

    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector if provided"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("Embedding must be numpy array")
            if v.ndim != 1:
                raise ValueError("Embedding must be 1-dimensional array")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
        return v

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata field"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field with default"""
        return self.metadata.get(key, default)

    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j storage"""
        result = {
            'id': self.id,
            'relation': self.relation,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'source_docs': self.source_docs,
            'entity_count': len(self.entities)
        }

        # Add metadata fields
        for key, value in self.metadata.items():
            result[f'meta_{key}'] = value

        # Handle embedding separately due to size
        if self.embedding is not None:
            result['has_embedding'] = True
            result['embedding_dim'] = len(self.embedding)
        else:
            result['has_embedding'] = False

        return result


class HippoNode(BaseModel):
    """
    Fast episodic memory node for recent interactions and temporary data.

    Named after hippocampus - brain region responsible for episodic memory.
    Designed for fast insertion/retrieval of user session data, recent queries,
    and temporary context that needs quick access but eventual consolidation.
    """

    id: str = Field(description="Unique node identifier")
    content: str = Field(description="Node content/data")
    episodic: bool = Field(default=True, description="Whether this is episodic memory")
    created: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_pattern: Optional[np.ndarray] = Field(default=None, description="Access pattern for PPR")

    # Session and context tracking
    session_id: Optional[str] = Field(default=None, description="User session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    context_type: str = Field(default="general", description="Type of context (query, response, etc.)")

    # Consolidation tracking
    consolidation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consolidated: bool = Field(default=False, description="Whether consolidated to semantic memory")
    consolidation_timestamp: Optional[datetime] = Field(default=None)

    # Performance tracking
    access_count: int = Field(default=1, ge=1, description="Number of accesses")
    relevance_decay: float = Field(default=1.0, ge=0.0, le=1.0, description="Time-based relevance decay")

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "id": "hippo_session_001",
                "content": "User asked about diabetes management options",
                "session_id": "user_session_12345",
                "context_type": "query"
            }
        }

    @validator('content')
    def validate_content(cls, v):
        """Ensure content is not empty"""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    @validator('access_pattern')
    def validate_access_pattern(cls, v):
        """Validate access pattern for Personalized PageRank"""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("Access pattern must be numpy array")
            if v.ndim != 1:
                raise ValueError("Access pattern must be 1-dimensional")
            if np.any(v < 0):
                raise ValueError("Access pattern values must be non-negative")
            # Normalize if not already
            if np.sum(v) > 0:
                v = v / np.sum(v)
        return v

    def update_access(self) -> None:
        """Update access tracking"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

        # Update relevance decay based on time since creation
        time_diff = (self.last_accessed - self.created).total_seconds()
        # Exponential decay with half-life of 1 hour
        self.relevance_decay = np.exp(-time_diff / 3600.0)

    def calculate_consolidation_score(self) -> float:
        """Calculate score for consolidating to semantic memory"""
        # Factors: access frequency, age, relevance
        age_hours = (datetime.utcnow() - self.created).total_seconds() / 3600.0

        # Higher score for frequently accessed, moderately aged content
        frequency_score = min(self.access_count / 10.0, 1.0)  # Normalize to [0,1]
        age_score = max(0.0, 1.0 - age_hours / 168.0)  # Decay over 1 week
        relevance_score = self.relevance_decay

        # Weighted combination
        self.consolidation_score = (
            0.4 * frequency_score +
            0.3 * age_score +
            0.3 * relevance_score
        )

        return self.consolidation_score

    def mark_consolidated(self) -> None:
        """Mark node as consolidated to semantic memory"""
        self.consolidated = True
        self.consolidation_timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {
            'id': self.id,
            'content': self.content,
            'episodic': self.episodic,
            'created': self.created.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'session_id': self.session_id,
            'user_id': self.user_id,
            'context_type': self.context_type,
            'consolidation_score': self.consolidation_score,
            'consolidated': self.consolidated,
            'access_count': self.access_count,
            'relevance_decay': self.relevance_decay
        }

        if self.consolidation_timestamp:
            result['consolidation_timestamp'] = self.consolidation_timestamp.isoformat()

        if self.access_pattern is not None:
            result['access_pattern'] = self.access_pattern.tolist()

        return result


# Utility functions for working with hypergraph structures

def create_medical_hyperedge(
    patient_id: str,
    medication_id: str,
    allergen_id: str,
    confidence: float,
    severity: str = "medium"
) -> Hyperedge:
    """Create a medical contraindication hyperedge"""
    return Hyperedge(
        entities=[patient_id, medication_id, allergen_id],
        relation="contraindicated_due_to_allergy",
        confidence=confidence,
        metadata={
            "domain": "medical",
            "severity": severity,
            "requires_review": severity in ["high", "critical"]
        }
    )


def create_query_response_hyperedge(
    query_id: str,
    document_ids: List[str],
    response_id: str,
    confidence: float
) -> Hyperedge:
    """Create a query-document-response hyperedge for RAG tracking"""
    entities = [query_id] + document_ids + [response_id]

    return Hyperedge(
        entities=entities,
        relation="rag_retrieval_response",
        confidence=confidence,
        metadata={
            "domain": "rag",
            "num_documents": len(document_ids),
            "response_type": "generated"
        }
    )


def create_session_hippo_node(
    session_id: str,
    user_id: str,
    content: str,
    context_type: str = "interaction"
) -> HippoNode:
    """Create a session-specific episodic memory node"""
    return HippoNode(
        id=f"hippo_{session_id}_{uuid.uuid4().hex[:8]}",
        content=content,
        session_id=session_id,
        user_id=user_id,
        context_type=context_type
    )
