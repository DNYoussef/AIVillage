"""
Analyze parameter distribution in Cogment system components.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "core", "agent_forge", "models"))


from cogment.core.config import CogmentConfig
from cogment.core.refinement_core import RefinementCore
from cogment.memory.gated_ltm import GatedLTMMemory


def count_module_params(module, name):
    """Count parameters in a module with breakdown."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: {total:,} total ({trainable:,} trainable)")
    return total, trainable


def analyze_refinement_core_params():
    """Analyze RefinementCore parameter breakdown."""
    print("Analyzing RefinementCore parameters...")

    config = CogmentConfig()
    print(f"Config: d_model={config.d_model}, vocab_size={config.vocab_size}, ltm_dim={config.ltm_dim}")

    refinement_core = RefinementCore(config)

    print("\nRefinementCore component breakdown:")

    # Encoder
    encoder_params, _ = count_module_params(refinement_core.encoder, "Encoder")

    # Memory gate
    memory_gate_params, _ = count_module_params(refinement_core.memory_gate, "MemoryGate")

    # Heads (the likely culprit)
    y_head_params, _ = count_module_params(refinement_core.y_head, "Y Head")
    delta_y_head_params, _ = count_module_params(refinement_core.delta_y_head, "Delta Y Head")
    halt_head_params, _ = count_module_params(refinement_core.halt_head, "Halt Head")

    # Other
    delta_scale_params = refinement_core.delta_scale.numel()
    print(f"Delta scale: {delta_scale_params:,}")

    total_params, _ = count_module_params(refinement_core, "Total RefinementCore")

    print("\nBreakdown:")
    print(f"  Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  MemoryGate: {memory_gate_params:,} ({memory_gate_params/total_params*100:.1f}%)")
    print(f"  Y Head: {y_head_params:,} ({y_head_params/total_params*100:.1f}%)")
    print(f"  Delta Y Head: {delta_y_head_params:,} ({delta_y_head_params/total_params*100:.1f}%)")
    print(f"  Halt Head: {halt_head_params:,} ({halt_head_params/total_params*100:.1f}%)")
    print(f"  Delta scale: {delta_scale_params:,}")

    # The issue: vocab heads are huge!
    vocab_head_params = y_head_params + delta_y_head_params
    print(f"\nVocabulary heads total: {vocab_head_params:,} ({vocab_head_params/total_params*100:.1f}%)")

    # Calculate expected size
    d_model = config.d_model
    vocab_size = config.vocab_size
    expected_y_head = d_model * vocab_size
    expected_delta_head = d_model * vocab_size
    expected_total = expected_y_head + expected_delta_head

    print("\nExpected vocab head sizes:")
    print(f"  Y head: {d_model} × {vocab_size} = {expected_y_head:,}")
    print(f"  Delta head: {d_model} × {vocab_size} = {expected_delta_head:,}")
    print(f"  Total: {expected_total:,}")

    return total_params


def analyze_memory_system_params():
    """Analyze GatedLTM parameter breakdown."""
    print("\nAnalyzing GatedLTM parameters...")

    config = CogmentConfig()

    memory_system = GatedLTMMemory(query_dim=config.d_model, memory_dim=config.ltm_dim, n_slots=config.ltm_capacity)

    print("\nGatedLTM component breakdown:")

    # Memory slots (non-trainable)
    memory_slots = config.ltm_capacity * config.ltm_dim * 2  # keys + values
    print(f"Memory slots: {memory_slots:,} (non-trainable)")

    # Components
    count_module_params(memory_system.cross_attention, "CrossAttention")
    count_module_params(memory_system.surprise_gate, "SurpriseGate")
    count_module_params(memory_system.memory_writer, "MemoryWriter")
    count_module_params(memory_system.query_projector, "QueryProjector")
    count_module_params(memory_system.output_projector, "OutputProjector")

    param_info = memory_system.count_parameters()
    print("\nTotal GatedLTM:")
    print(f"  Trainable: {param_info['trainable']:,}")
    print(f"  Memory slots: {param_info['memory_slots']:,}")
    print(f"  Total: {param_info['total']:,}")

    return param_info["total"]


def suggest_optimizations():
    """Suggest parameter optimizations."""
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION SUGGESTIONS")
    print("=" * 60)

    config = CogmentConfig()

    # Current vocab head size
    current_vocab_params = 2 * config.d_model * config.vocab_size
    print(f"Current vocab heads: {current_vocab_params:,} parameters")

    # Option 1: Reduce vocab size
    reduced_vocab = 8000
    reduced_vocab_params = 2 * config.d_model * reduced_vocab
    print(f"Option 1 - Reduce vocab to {reduced_vocab}: {reduced_vocab_params:,} parameters")
    print(f"  Savings: {current_vocab_params - reduced_vocab_params:,}")

    # Option 2: Smaller d_model
    smaller_d_model = 256
    smaller_model_vocab_params = 2 * smaller_d_model * config.vocab_size
    print(f"Option 2 - Reduce d_model to {smaller_d_model}: {smaller_model_vocab_params:,} parameters")
    print(f"  Savings: {current_vocab_params - smaller_model_vocab_params:,}")

    # Option 3: Tied heads (Agent 3 responsibility)
    tied_heads_params = config.d_model * config.vocab_size  # Only one head
    print(f"Option 3 - Tie Y and Delta heads: {tied_heads_params:,} parameters")
    print(f"  Savings: {current_vocab_params - tied_heads_params:,}")

    # Option 4: Factorized heads
    bottleneck_dim = 128
    factorized_params = 2 * (config.d_model * bottleneck_dim + bottleneck_dim * config.vocab_size)
    print(f"Option 4 - Factorized heads (bottleneck={bottleneck_dim}): {factorized_params:,} parameters")
    print(f"  Savings: {current_vocab_params - factorized_params:,}")

    print("\nRECOMMENDATION: Agent 3 should implement Option 3 or 4 to stay within budget!")


def calculate_budget_allocation():
    """Calculate parameter budget allocation."""
    print("\n" + "=" * 60)
    print("PARAMETER BUDGET ALLOCATION")
    print("=" * 60)

    total_budget = 25_000_000  # 25M total for Cogment
    current_usage = analyze_refinement_core_params() + analyze_memory_system_params()

    print(f"\nCurrent usage: {current_usage:,}")
    print(f"Total budget: {total_budget:,}")
    print(f"Remaining: {total_budget - current_usage:,}")
    print(f"Usage: {current_usage / total_budget * 100:.1f}%")

    if current_usage > total_budget:
        print(f"⚠️  OVER BUDGET by {current_usage - total_budget:,} parameters!")
    else:
        print(f"✅ Within budget with {total_budget - current_usage:,} parameters remaining")


if __name__ == "__main__":
    analyze_refinement_core_params()
    analyze_memory_system_params()
    suggest_optimizations()
    calculate_budget_allocation()
