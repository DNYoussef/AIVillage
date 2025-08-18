"""
Creativity subsystem for HyperRAG - Non-obvious path discovery and insight generation.

This module provides creative reasoning capabilities including insight discovery,
analogical reasoning, and creative connection exploration.
"""

from .insight_engine import (
    CreativeAnalogy,
    CreativeInsight,
    CreativityEngine,
    CreativityMethod,
    InsightDiscoveryResult,
    InsightType,
)

__all__ = [
    "CreativityEngine",
    "CreativeInsight",
    "CreativeAnalogy",
    "InsightDiscoveryResult",
    "InsightType",
    "CreativityMethod",
]
