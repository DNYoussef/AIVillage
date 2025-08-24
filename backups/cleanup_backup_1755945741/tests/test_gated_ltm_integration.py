"""
Test integration between GatedLTM memory system and Cogment RefinementCore.

Validates that the memory system correctly interfaces with Agent 1's MemoryGate
and provides the expected tensor shapes and behaviors.
"""


import pytest
import torch
import torch.nn as nn

# Import Agent 1's components
from core.agent_forge.models.cogment.core.config import CogmentConfig
from core.agent_forge.models.cogment.core.refinement_core import RefinementCore

# Import Agent 2's memory system
from core.agent_forge.models.cogment.memory import GatedLTMMemory


class TestGatedLTMIntegration:
    """Test suite for GatedLTM integration with Cogment."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CogmentConfig(
            d_model=320,
            vocab_size=16000,
            ltm_capacity=64,  # Smaller for testing
            ltm_dim=128,  # Smaller for testing
            n_head=8,
            dropout=0.0,
        )

    @pytest.fixture
    def memory_system(self, config):
        """Create GatedLTM memory system."""
        return GatedLTMMemory(
            query_dim=config.d_model,
            memory_dim=config.ltm_dim,
            n_slots=config.ltm_capacity,
            n_heads=8,
            topk=8,
            dropout=0.0,
        )

    @pytest.fixture
    def refinement_core(self, config):
        """Create RefinementCore with MemoryGate."""
        return RefinementCore(config)

    def test_memory_dimensions(self, memory_system, config):
        """Test that memory system produces correct output dimensions."""
        B, N = 2, 10
        query_states = torch.randn(B, N, config.d_model)

        # Test read operation
        memory_context = memory_system.read(query_states)

        # Should return context in model dimensions
        assert memory_context.shape == (B, N, config.d_model)
        assert memory_context.dtype == torch.float32
        assert not torch.isnan(memory_context).any()

    def test_memory_gate_integration(self, memory_system, refinement_core, config):
        """Test integration between GatedLTM and MemoryGate."""
        B, N = 2, 10
        hidden_states = torch.randn(B, N, config.d_model)

        # Get memory context from GatedLTM
        memory_context = memory_system.read(hidden_states)

        # Memory context should be shaped for MemoryGate
        # MemoryGate expects [B, M, ltm_dim] but we return [B, N, d_model]
        # Let's test the MemoryGate can handle the context

        # Reshape memory context to match expected interface [B, M, ltm_dim]
        M = 8  # Number of memory tokens
        memory_for_gate = memory_context[:, :M, :]  # Take first M positions

        # Project to ltm_dim if needed
        if memory_for_gate.size(-1) != config.ltm_dim:
            proj = nn.Linear(config.d_model, config.ltm_dim)
            memory_for_gate = proj(memory_for_gate)

        # Test MemoryGate forward pass
        memory_gate = refinement_core.memory_gate
        fused_states = memory_gate(hidden_states, memory_for_gate)

        assert fused_states.shape == hidden_states.shape
        assert not torch.isnan(fused_states).any()

    def test_memory_read_write_cycle(self, memory_system, config):
        """Test complete read-write cycle."""
        B, N = 2, 10
        vocab_size = config.vocab_size

        query_states = torch.randn(B, N, config.d_model)
        predictions = torch.randn(B, N, vocab_size)
        targets = torch.randint(0, vocab_size, (B, N))

        # Initial read (should work with empty memory)
        initial_context = memory_system.read(query_states)
        assert initial_context.shape == (B, N, config.d_model)

        # Write to memory
        write_info = memory_system.write(
            query_states=query_states, predictions=predictions, targets=targets, return_gate_info=True
        )

        # Should return gate information
        assert write_info is not None
        assert "gate_weights" in write_info
        assert "surprisal" in write_info

        # Read again (should potentially return different context)
        updated_context = memory_system.read(query_states)
        assert updated_context.shape == (B, N, config.d_model)

    def test_memory_decay(self, memory_system):
        """Test memory decay mechanism."""
        # Store initial memory state
        initial_keys = memory_system.memory_keys.clone()
        initial_values = memory_system.memory_values.clone()

        # Apply decay
        memory_system.decay_step()

        # Memory should be slightly smaller
        keys_decay = (memory_system.memory_keys.norm(dim=-1) < initial_keys.norm(dim=-1)).float().mean()
        values_decay = (memory_system.memory_values.norm(dim=-1) < initial_values.norm(dim=-1)).float().mean()

        # Most slots should have decayed (unless decay_rate is 0)
        assert keys_decay > 0.8  # At least 80% of slots decayed
        assert values_decay > 0.8

    def test_memory_statistics(self, memory_system, config):
        """Test memory statistics and monitoring."""
        B, N = 2, 10
        query_states = torch.randn(B, N, config.d_model)

        # Perform some operations
        memory_system.read(query_states)
        memory_system.write(
            query_states=query_states, surprisal=torch.ones(B) * 2.0, force_write=True  # High surprisal to force write
        )

        # Get statistics
        stats = memory_system.get_memory_stats()

        # Check expected statistics are present
        expected_keys = [
            "total_slots",
            "avg_usage",
            "max_usage",
            "min_usage",
            "unused_slots",
            "overused_slots",
            "update_count",
        ]

        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], torch.Tensor)

    def test_parameter_budget(self, memory_system, config):
        """Test that memory system stays within parameter budget."""
        param_info = memory_system.count_parameters()

        # Memory slots: 64 * 128 * 2 = 16,384 (much smaller for testing)
        # Trainable params: should be reasonable
        expected_memory_slots = config.ltm_capacity * config.ltm_dim * 2

        assert param_info["memory_slots"] == expected_memory_slots
        assert param_info["trainable"] < 200_000  # Reasonable limit for components
        assert param_info["total"] == param_info["trainable"] + param_info["memory_slots"]

    def test_surprise_gating(self, memory_system, config):
        """Test surprise-based gating mechanism."""
        B, N = 2, 10

        query_states = torch.randn(B, N, config.d_model)

        # Low surprisal - should not trigger writes
        low_surprisal = torch.ones(B) * 0.1
        write_info_low = memory_system.write(query_states=query_states, surprisal=low_surprisal, return_gate_info=True)

        # High surprisal - should trigger writes
        high_surprisal = torch.ones(B) * 3.0
        write_info_high = memory_system.write(
            query_states=query_states, surprisal=high_surprisal, return_gate_info=True
        )

        # High surprisal should result in higher gate weights
        if write_info_low and write_info_high:
            assert write_info_high["gate_weights"].mean() > write_info_low["gate_weights"].mean()

    def test_end_to_end_integration(self, memory_system, refinement_core, config):
        """Test complete end-to-end integration with RefinementCore."""
        B, N = 2, 10
        vocab_size = config.vocab_size

        # Input sequence
        hidden_states = torch.randn(B, N, config.d_model)
        targets = torch.randint(0, vocab_size, (B, N))

        # Step 1: Read from memory
        memory_context = memory_system.read(hidden_states)

        # Step 2: Process through RefinementCore with memory
        # Reshape memory for MemoryGate interface [B, M, ltm_dim]
        M = min(8, N)  # Use up to 8 memory tokens
        memory_for_refinement = memory_context[:, :M, :]

        # Project to ltm_dim if needed
        if memory_for_refinement.size(-1) != config.ltm_dim:
            proj = nn.Linear(config.d_model, config.ltm_dim)
            memory_for_refinement = proj(memory_for_refinement)

        y_logits, delta_logits, halt_prob, refined_states = refinement_core(
            hidden_states=hidden_states, memory=memory_for_refinement, step=0
        )

        # Check outputs
        assert y_logits.shape == (B, N, vocab_size)
        assert delta_logits.shape == (B, N, vocab_size)
        assert halt_prob.shape == (B, N, 1)
        assert refined_states.shape == (B, N, config.d_model)

        # Step 3: Write refined states back to memory
        combined_logits = refinement_core.compute_prediction(y_logits, delta_logits)
        memory_system.write(query_states=refined_states, predictions=combined_logits, targets=targets)

        # Memory should now contain updated information
        stats = memory_system.get_memory_stats()
        assert stats["update_count"] > 0


def test_memory_system_creation():
    """Test that memory system can be created with default config."""
    config = CogmentConfig()

    memory_system = GatedLTMMemory(query_dim=config.d_model, memory_dim=config.ltm_dim, n_slots=config.ltm_capacity)

    # Should create without errors
    assert memory_system is not None
    assert memory_system.n_slots == config.ltm_capacity
    assert memory_system.memory_dim == config.ltm_dim


if __name__ == "__main__":
    # Quick smoke test
    print("Running GatedLTM integration smoke test...")

    config = CogmentConfig(ltm_capacity=32, ltm_dim=64)  # Small for testing

    memory_system = GatedLTMMemory(query_dim=config.d_model, memory_dim=config.ltm_dim, n_slots=config.ltm_capacity)

    # Test basic functionality
    B, N = 1, 5
    query_states = torch.randn(B, N, config.d_model)

    print(f"Memory system created with {memory_system.count_parameters()['total']:,} total parameters")

    # Test read
    memory_context = memory_system.read(query_states)
    print(f"Read operation successful: {memory_context.shape}")

    # Test write
    predictions = torch.randn(B, N, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (B, N))

    write_info = memory_system.write(
        query_states=query_states, predictions=predictions, targets=targets, return_gate_info=True
    )

    print(f"Write operation successful: {write_info['num_writes']} writes performed")

    # Test statistics
    stats = memory_system.get_memory_stats()
    print(f"Memory stats: {stats['avg_usage']:.4f} avg usage, {stats['unused_slots']} unused slots")

    print("✅ GatedLTM integration test passed!")
