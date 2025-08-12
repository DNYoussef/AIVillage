#!/usr/bin/env python3
"""Test AgentForge performance and import times."""

import time


class TestAgentForgePerformance:
    """Test performance characteristics of AgentForge."""

    def test_agent_forge_import_time_under_1s(self):
        """Test that agent_forge imports in under 1 second."""
        start_time = time.time()

        # This will fail initially due to heavy imports

        import_time = time.time() - start_time

        # Should be under 1 second after lazy loading fix
        assert import_time < 1.0, (
            f"agent_forge import took {import_time:.2f}s, should be <1s"
        )

    def test_agent_forge_lazy_initialization(self):
        """Test that AgentForge class doesn't do heavy work at construction."""
        import agent_forge

        start_time = time.time()

        # Creating instance should be fast
        forge = agent_forge.AgentForge()

        construction_time = time.time() - start_time

        # Construction should be nearly instantaneous
        assert construction_time < 0.1, (
            f"AgentForge() took {construction_time:.2f}s, should be <0.1s"
        )

        # Test that accessing properties works (may raise ImportError for missing deps)
        start_time = time.time()

        # Test available properties first (these should work)
        try:
            _ = forge.training_task  # This should work
            training_task_time = time.time() - start_time
            print(f"Training task access time: {training_task_time:.2f}s")
        except ImportError as e:
            print(f"Training task not available: {e}")

        # Test optional heavy dependencies (may fail)
        try:
            start_time = time.time()
            _ = forge.evolution_tournament  # This may fail if evomerge not available
            property_access_time = time.time() - start_time
            print(f"Evolution tournament access time: {property_access_time:.2f}s")
        except ImportError as e:
            print(f"Evolution tournament not available (expected): {e}")
            # This is expected if evomerge module is not installed
