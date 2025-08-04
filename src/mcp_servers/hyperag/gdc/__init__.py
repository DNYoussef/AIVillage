"""HypeRAG Graph-Doctor Constraint System.

Detects forbidden sub-graph patterns (Graph Denial Constraints) and outputs
structured Violation objects for the Innovator Repair flow.
"""

from .extractor import GDCExtractor
from .registry import GDC_REGISTRY, load_gdc_registry
from .specs import GDCSpec, Violation

__all__ = ["GDC_REGISTRY", "GDCExtractor", "GDCSpec", "Violation", "load_gdc_registry"]
