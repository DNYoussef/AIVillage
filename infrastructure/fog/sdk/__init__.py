"""
AIVillage Fog Computing SDK

Python client library for interacting with the AIVillage Fog Gateway.
Provides high-level interfaces for job submission, sandbox management,
and usage tracking.
"""

from .python.fog_client import FogClient

__version__ = "1.0.0"
__all__ = ["FogClient"]
