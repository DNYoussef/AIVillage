"""Simple path explainer POC using Neo4j (mocked)."""
from __future__ import annotations

from typing import Dict, List


def explain_path(start: str, end: str) -> Dict[str, List[str]]:
    """Return a mocked shortest path between two nodes."""
    # In a real system this would query Neo4j.
    return {"path": [start, "intermediate", end]}
