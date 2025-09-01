"""
Test integration between Agent 2's GatedLTM and Agent 1's RefinementCore.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "core", "agent_forge", "models"))

# Import Agent 1's components
from cogment.core.config import CogmentConfig
from cogment.core.refinement_core import MemoryGate, RefinementCore

# Import Agent 2's memory system
from cogment.memory.gated_ltm import GatedLTMMemory
import torch
import torch.nn as nn


def test_agent_integration():
    """Test integration between GatedLTM (Agent 2) and RefinementCore (Agent 1)."""
    print("Testing Agent 1 + Agent 2 Integration...")

    # Create config
    config = CogmentConfig(d_model=320, ltm_capacity=64, ltm_dim=512, vocab_size=16000)  # Use full config value

    print(f"Config: d_model={config.d_model}, ltm_dim={config.ltm_dim}")

    # Create Agent 1's RefinementCore
    refinement_core = RefinementCore(config)
    refinement_params = refinement_core.count_parameters()
    print(f"RefinementCore parameters: {refinement_params:,}")

    # Create Agent 2's GatedLTM Memory
    memory_system = GatedLTMMemory(
        query_dim=config.d_model, memory_dim=config.ltm_dim, n_slots=config.ltm_capacity, n_heads=8, topk=8
    )
    memory_params = memory_system.count_parameters()
    print(f"GatedLTM parameters: {memory_params['total']:,} total ({memory_params['trainable']:,} trainable)")

    total_params = refinement_params + memory_params["total"]
    print(f"Combined system parameters: {total_params:,}")

    # Test data
    B, N = 2, 10
    hidden_states = torch.randn(B, N, config.d_model)
    targets = torch.randint(0, config.vocab_size, (B, N))

    print("\nStep 1: Read from memory...")
    # Step 1: Read from memory (Agent 2)
    memory_context = memory_system.read(hidden_states)
    print(f"Memory context shape: {memory_context.shape}")

    # Convert memory context to the format expected by MemoryGate
    # MemoryGate expects [B, M, ltm_dim] but we get [B, N, d_model]
    # We need to create memory representations in ltm_dim space
    M = 8  # Number of memory tokens to use

    print("\nStep 2: Process through RefinementCore with memory...")

    # Option 1: Use a subset and project to ltm_dim
    memory_subset = memory_context[:, :M, :]  # [B, M, d_model]

    # Project to ltm_dim space
    memory_proj = nn.Linear(config.d_model, config.ltm_dim)
    memory_for_refinement = memory_proj(memory_subset)  # [B, M, ltm_dim]

    print(f"Memory for refinement shape: {memory_for_refinement.shape}")

    # Step 2: Process through RefinementCore (Agent 1)
    y_logits, delta_logits, halt_prob, refined_states = refinement_core(
        hidden_states=hidden_states, memory=memory_for_refinement, step=0
    )

    print("RefinementCore outputs:")
    print(f"  y_logits: {y_logits.shape}")
    print(f"  delta_logits: {delta_logits.shape}")
    print(f"  halt_prob: {halt_prob.shape}")
    print(f"  refined_states: {refined_states.shape}")

    # Combine predictions
    combined_logits = refinement_core.compute_prediction(y_logits, delta_logits)
    print(f"  combined_logits: {combined_logits.shape}")

    print("\nStep 3: Write refined states back to memory...")
    # Step 3: Write back to memory (Agent 2)
    write_info = memory_system.write(
        query_states=refined_states, predictions=combined_logits, targets=targets, return_gate_info=True
    )

    if write_info:
        print(f"Write successful: {write_info['num_writes']} slots updated")
        print(f"Gate weights: {write_info['gate_weights'].mean():.4f}")
        print(f"Surprisal: {write_info['surprisal'].mean():.4f}")
    else:
        print("No writes performed")

    print("\nStep 4: Test complete cycle...")
    # Test a complete read-process-write cycle
    for step in range(3):
        print(f"\nCycle {step + 1}:")

        # Read current memory
        memory_context = memory_system.read(hidden_states)

        # Prepare memory for refinement
        memory_subset = memory_context[:, :M, :]
        memory_for_refinement = memory_proj(memory_subset)

        # Process through refinement
        y_logits, delta_logits, halt_prob, refined_states = refinement_core(
            hidden_states=hidden_states, memory=memory_for_refinement, step=step
        )

        # Write back to memory
        combined_logits = refinement_core.compute_prediction(y_logits, delta_logits)
        write_info = memory_system.write(
            query_states=refined_states, predictions=combined_logits, targets=targets, return_gate_info=True
        )

        if write_info:
            print(f"  Writes: {write_info['num_writes']}, Surprisal: {write_info['surprisal'].mean():.4f}")

        # Update hidden states for next iteration
        hidden_states = refined_states

    # Final statistics
    stats = memory_system.get_memory_stats()
    print("\nFinal memory statistics:")
    print(f"  Total updates: {stats['update_count']}")
    print(f"  Average usage: {stats['avg_usage']:.4f}")
    print(f"  Unused slots: {stats['unused_slots']}")

    print("\nIntegration test successful!")
    print(f"Total system parameters: {total_params:,}")

    # Parameter budget check
    budget = 1_400_000  # 1.4M parameter budget
    if total_params <= budget:
        print(f"Within parameter budget: {total_params:,} / {budget:,} ({total_params/budget*100:.1f}%)")
    else:
        print(f"Exceeds parameter budget: {total_params:,} / {budget:,} ({total_params/budget*100:.1f}%)")

    return True


def test_memory_gate_directly():
    """Test MemoryGate component directly."""
    print("\nTesting MemoryGate directly...")

    config = CogmentConfig()
    memory_gate = MemoryGate(d_model=config.d_model, memory_dim=config.ltm_dim)

    B, N, M = 2, 10, 8
    hidden_states = torch.randn(B, N, config.d_model)
    memory = torch.randn(B, M, config.ltm_dim)

    # Test without memory
    output_no_mem = memory_gate(hidden_states, None)
    print(f"Output without memory: {output_no_mem.shape}")
    assert torch.allclose(output_no_mem, hidden_states, rtol=1e-5)

    # Test with memory
    output_with_mem = memory_gate(hidden_states, memory)
    print(f"Output with memory: {output_with_mem.shape}")
    assert output_with_mem.shape == hidden_states.shape
    assert not torch.allclose(output_with_mem, hidden_states)

    print("MemoryGate test successful!")


if __name__ == "__main__":
    test_memory_gate_directly()
    test_agent_integration()
