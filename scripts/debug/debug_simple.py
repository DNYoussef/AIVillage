#!/usr/bin/env python3
"""
Simplest possible test to isolate the issue
"""

import torch
import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.resolve()
cognate_path = script_dir / "core" / "agent_forge" / "phases" / "cognate_pretrain"
packages_path = script_dir / "packages"
sys.path.insert(0, str(cognate_path))
sys.path.insert(0, str(packages_path))

try:
    print("Step 1: Import UnifiedRefinerOrchestrator directly")
    from agent_forge.models.cognate.unified_refiner.orchestrator import UnifiedRefinerOrchestrator
    CognateOrchestrator = UnifiedRefinerOrchestrator
    
    print("Step 2: Create mock model")
    class MockModel:
        def __init__(self):
            self.config = self.MockConfig()
        
        class MockConfig:
            d_model = 216
            d_mem = 216
            n_heads = 4
            max_seq_len = 2048
    
    print("Step 3: Create mock tokenizer")
    class MockTokenizer:
        vocab_size = 32000
        bos_token_id = 1
    
    print("Step 4: Create mock memory bank")
    class MockMemoryBank:
        pass
    
    print("Step 5: Create mock config")  
    class MockConfig:
        max_steps_train = 2
        max_steps_infer = 1
        act_threshold = 0.99
        ltm_read_policy = 216
        ltm_write_policy = 216
    
    print("Step 6: Initialize orchestrator")
    model = MockModel()
    tokenizer = MockTokenizer()
    memory_bank = MockMemoryBank()
    config = MockConfig()
    device = torch.device('cpu')
    
    orchestrator = CognateOrchestrator(
        model=model,
        tokenizer=tokenizer,
        memory_bank=memory_bank,
        config=config,
        device=device
    )
    
    print("SUCCESS: Orchestrator created without error")
    
except Exception as e:
    print(f"Error at step: {e}")
    import traceback
    traceback.print_exc()