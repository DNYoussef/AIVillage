"""
HypeRAG Graph-Doctor Constraint System

Detects forbidden sub-graph patterns (Graph Denial Constraints) and outputs
structured Violation objects for the Innovator Repair flow.
"""

from .specs import GDCSpec, Violation
from .registry import GDC_REGISTRY, load_gdc_registry
from .extractor import GDCExtractor

__all__ = [
    "GDCSpec",
    "Violation",
    "GDC_REGISTRY",
    "load_gdc_registry",
    "GDCExtractor"
]
