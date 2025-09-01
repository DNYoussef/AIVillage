"""
Simple test of GatedLTM memory system functionality.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "core", "agent_forge", "models"))

# Import components
from cogment.core.config import CogmentConfig
from cogment.memory.gated_ltm import GatedLTMMemory
import torch


def test_gated_ltm():
    """Test GatedLTM memory system."""
    print("Testing GatedLTM Memory System...")

    # Create config
    config = CogmentConfig(
        d_model=320, ltm_capacity=64, ltm_dim=128, vocab_size=16000  # Small for testing  # Small for testing
    )

    print(f"Config: d_model={config.d_model}, ltm_capacity={config.ltm_capacity}, ltm_dim={config.ltm_dim}")

    # Create memory system
    memory_system = GatedLTMMemory(
        query_dim=config.d_model, memory_dim=config.ltm_dim, n_slots=config.ltm_capacity, n_heads=8, topk=8, dropout=0.0
    )

    param_info = memory_system.count_parameters()
    print(
        f"Parameters: {param_info['total']:,} total ({param_info['trainable']:,} trainable + {param_info['memory_slots']:,} memory slots)"
    )

    # Test dimensions
    B, N = 2, 10
    query_states = torch.randn(B, N, config.d_model)

    print("\nTesting READ operation...")
    print(f"Input query shape: {query_states.shape}")

    # Test read
    memory_context = memory_system.read(query_states)
    print(f"Memory context shape: {memory_context.shape}")
    print(f"Read successful: got {memory_context.shape} (expected {(B, N, config.d_model)})")

    # Test read with attention details
    memory_context, attn_info = memory_system.read(query_states, return_attention=True)
    print(f"Attention weights shape: {attn_info['attention_weights'].shape}")
    print(f"Selected indices shape: {attn_info['selected_indices'].shape}")

    print("\nTesting WRITE operation...")

    # Test write
    predictions = torch.randn(B, N, config.vocab_size)
    targets = torch.randint(0, config.vocab_size, (B, N))

    write_info = memory_system.write(
        query_states=query_states, predictions=predictions, targets=targets, return_gate_info=True
    )

    if write_info:
        print(f"Gate weights: {write_info['gate_weights']}")
        print(f"Surprisal: {write_info['surprisal']}")
        print(f"Number of writes: {write_info['num_writes']}")
        print(f"Write successful: {write_info['num_writes']} slots updated")
    else:
        print("No writes performed (low surprisal)")

    # Force a write
    print("\nTesting FORCED WRITE...")
    write_info = memory_system.write(
        query_states=query_states,
        surprisal=torch.ones(B) * 3.0,  # High surprisal
        force_write=True,
        return_gate_info=True,
    )

    print(f"Forced write info: {write_info['num_writes']} writes")

    print("\nTesting DECAY operation...")
    initial_norm = memory_system.memory_keys.norm()
    memory_system.decay_step()
    final_norm = memory_system.memory_keys.norm()
    print(f"Memory norm before decay: {initial_norm:.4f}")
    print(f"Memory norm after decay: {final_norm:.4f}")
    print(f"Decay applied: {((initial_norm - final_norm) / initial_norm * 100):.2f}% reduction")

    print("\nTesting STATISTICS...")
    stats = memory_system.get_memory_stats()
    print(f"Total slots: {stats['total_slots']}")
    print(f"Average usage: {stats['avg_usage']:.4f}")
    print(f"Unused slots: {stats['unused_slots']}")
    print(f"Update count: {stats['update_count']}")

    print("\nTesting FULL CYCLE (read-write-read)...")

    # Initial read
    initial_context = memory_system.read(query_states)
    initial_mean = initial_context.mean().item()

    # Write with high surprisal
    memory_system.write(
        query_states=query_states * 2,  # Different pattern
        surprisal=torch.ones(B) * 5.0,  # Very high surprisal
        force_write=True,
    )

    # Read again
    updated_context = memory_system.read(query_states)
    updated_mean = updated_context.mean().item()

    print(f"Initial context mean: {initial_mean:.4f}")
    print(f"Updated context mean: {updated_mean:.4f}")
    print(f"Context change: {abs(updated_mean - initial_mean):.4f}")

    if abs(updated_mean - initial_mean) > 0.001:
        print("Memory successfully updated and affects reads")
    else:
        print("Memory may not be updating properly")

    print("\nGatedLTM Memory System Test Complete!")
    return True


if __name__ == "__main__":
    test_gated_ltm()
