#!/usr/bin/env python3
"""
Debug script to isolate the torch.empty() error in the orchestrator.
"""

from pathlib import Path
import sys

import torch

# Add to Python path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent.parent.parent
cognate_path = project_root / "packages" / "agent_forge" / "models" / "cognate"
sys.path.insert(0, str(cognate_path))

try:
    from refiner_core import CognateConfig, CognateRefiner
    from unified_refiner import OrchestrationConfig, UnifiedRefinerOrchestrator, create_memory_bank

    print("SUCCESS: Successfully imported Cognate components")

    # Create minimal config
    config = CognateConfig(
        vocab_size=1000,  # Smaller for testing
        d_model=216,
        n_layers=2,  # Much smaller for faster testing
        n_heads=4,
        max_seq_len=128,
    )

    # Create model
    print("Creating model...")
    model = CognateRefiner(config)
    # Count parameters manually
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SUCCESS: Model created with {total_params:,} parameters")

    # Create memory bank
    print("Creating memory bank...")
    memory_bank = create_memory_bank(capacity=1000, d_mem=216, device=torch.device("cpu"))
    print("SUCCESS: Memory bank created")

    # Create orchestrator config
    print("Creating orchestrator config...")
    orch_config = OrchestrationConfig(
        t_max_train=4,
        t_min_train=2,
        t_max_infer=2,
        t_min_infer=1,
        lambda_act=0.1,
        alpha_read=0.05,
        beta_write=0.05,
        gamma_comp=0.02,
    )
    print("SUCCESS: Orchestrator config created")

    # Create tokenizer (mock)
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1

        def encode(self, text):
            return [1, 2, 3, 4]

        def decode(self, ids):
            return "test text"

    tokenizer = MockTokenizer()
    print("SUCCESS: Mock tokenizer created")

    # Create orchestrator
    print("Creating orchestrator...")
    orchestrator = UnifiedRefinerOrchestrator(
        model=model, tokenizer=tokenizer, memory_bank=memory_bank, config=orch_config, device=torch.device("cpu")
    )
    print("SUCCESS: Orchestrator created successfully")

    # Create minimal batch
    print("Creating test batch...")
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 64)),  # batch_size=2, seq_len=64
        "labels": torch.randint(0, 1000, (2, 64)),
        "attention_mask": torch.ones(2, 64),
    }
    print("SUCCESS: Test batch created")

    # Try the train_batch method
    print("Testing orchestrator.train_batch()...")
    try:
        outputs = orchestrator.train_batch(batch, target=batch["labels"], max_steps_override=2)
        print("SUCCESS: train_batch() succeeded!")
        print(f"Output keys: {outputs.keys() if outputs else 'None'}")
    except Exception as e:
        print(f"ERROR: train_batch() failed: {e}")
        import traceback

        print("Full traceback:")
        traceback.print_exc()

except Exception as e:
    print(f"ERROR: Import or setup failed: {e}")
    import traceback

    print("Full traceback:")
    traceback.print_exc()
