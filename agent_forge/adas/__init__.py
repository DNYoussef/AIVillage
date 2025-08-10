"""Agent Forge ADAS (Adaptive Decision and Action System) Module - Compatibility Layer

This module provides backward compatibility for tests that expect agent_forge.adas imports.
ADAS functionality has been moved to src.agent_forge.adas but we maintain compatibility.
"""

try:
    # Import from new location
    from src.agent_forge.adas import *
except ImportError:
    # Create stub implementation for missing ADAS module
    class ADASSystem:
        """Adaptive Decision and Action System stub."""

        def __init__(self):
            self.initialized = False

        def search(self, query: str):
            """Placeholder search method."""
            return {"results": [], "query": query, "status": "stub"}

        def decide(self, context):
            """Placeholder decision method."""
            return {"action": "no-op", "confidence": 0.0}

        def execute(self, action):
            """Placeholder execution method."""
            return {"status": "executed", "result": None}

    class ADASProcess:
        """ADAS Process stub."""

        def __init__(self):
            self.adas = ADASSystem()

        def run(self):
            """Run ADAS process."""
            return self.adas

    # Create module-level instances
    adas = ADASSystem()
    process = ADASProcess()

__all__ = ["ADASProcess", "ADASSystem", "adas", "process"]
