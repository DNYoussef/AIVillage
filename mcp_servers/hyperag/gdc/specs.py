"""
Graph Denial Constraint (GDC) Specifications

Defines the data structures for representing GDC rules and violations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from uuid import uuid4


@dataclass
class GDCSpec:
    """Specification for a Graph Denial Constraint rule"""

    id: str                          # Unique GDC identifier (e.g., "GDC_CONFIDENCE_VIOLATION")
    description: str                 # Human-readable description
    cypher: str                     # Cypher query to detect violations
    severity: str                   # "low" | "medium" | "high"
    suggested_action: str           # Repair action identifier
    category: str = "general"       # Constraint category
    enabled: bool = True            # Whether this GDC is active
    performance_hint: str = ""      # Query optimization hints

    def __post_init__(self):
        """Validate GDC specification"""
        if self.severity not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid severity: {self.severity}")

        if not self.id.startswith("GDC_"):
            raise ValueError(f"GDC ID must start with 'GDC_': {self.id}")


@dataclass
class Violation:
    """Represents a detected Graph Denial Constraint violation"""

    violation_id: str = field(default_factory=lambda: str(uuid4()))
    gdc_id: str = ""                                    # GDC rule that was violated
    nodes: List[Dict[str, Any]] = field(default_factory=list)      # Violating nodes
    edges: List[Dict[str, Any]] = field(default_factory=list)      # Violating edges
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # Relationship data
    severity: str = "medium"                            # Inherited from GDCSpec
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)         # Additional context
    suggested_repair: str = ""                          # Repair action identifier
    confidence_score: float = 1.0                      # Detection confidence [0,1]
    graph_context: Dict[str, Any] = field(default_factory=dict)    # Surrounding graph info

    def __post_init__(self):
        """Validate violation data"""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(f"Confidence score must be in [0,1]: {self.confidence_score}")

        if self.severity not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid severity: {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary for JSON serialization"""
        return {
            "violation_id": self.violation_id,
            "gdc_id": self.gdc_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "relationships": self.relationships,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "metadata": self.metadata,
            "suggested_repair": self.suggested_repair,
            "confidence_score": self.confidence_score,
            "graph_context": self.graph_context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Violation":
        """Create violation from dictionary"""
        data = data.copy()
        if "detected_at" in data and isinstance(data["detected_at"], str):
            data["detected_at"] = datetime.fromisoformat(data["detected_at"])
        return cls(**data)

    def get_affected_node_ids(self) -> List[str]:
        """Extract node IDs from violation"""
        node_ids = []
        for node in self.nodes:
            if "id" in node:
                node_ids.append(node["id"])
        return node_ids

    def get_affected_edge_ids(self) -> List[str]:
        """Extract edge IDs from violation"""
        edge_ids = []
        for edge in self.edges:
            if "id" in edge:
                edge_ids.append(edge["id"])
        return edge_ids

    def add_context(self, key: str, value: Any) -> None:
        """Add contextual information to violation"""
        self.metadata[key] = value
