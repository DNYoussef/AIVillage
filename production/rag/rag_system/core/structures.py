# rag_system/core/structures.py

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

@dataclass(frozen=True)
class BayesianNode:
    id: str
    content: str
    probability: float
    uncertainty: float
    timestamp: datetime
    version: int

@dataclass(frozen=True)
class VectorEntry:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int

@dataclass(frozen=True)
class RetrievalResult:
    id: str
    content: str
    score: float
    uncertainty: float
    timestamp: datetime
    version: int

# Add a new structure for representing a plan
@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    strategy: str
    filters: Dict[str, Any]
    use_linearization: bool
    timestamp: datetime
    version: int
