#!/usr/bin/env python3
"""
Test with real OrchestrationConfig
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
    print("Step 1: Import components")
    from agent_forge.models.cognate.unified_refiner.orchestrator import UnifiedRefinerOrchestrator, OrchestrationConfig
    
    print("Step 2: Create real OrchestrationConfig")
    config = OrchestrationConfig()
    
    print("Step 3: Create mock model")
    class MockModel:
        def __init__(self):
            self.config = self.MockConfig()
        
        class MockConfig:
            d_model = 216
            d_mem = 216
            n_heads = 4
            max_seq_len = 2048
    
    print("Step 4: Create mock tokenizer")
    class MockTokenizer:
        vocab_size = 32000
        bos_token_id = 1
    
    print("Step 5: Create mock memory bank")
    class MockMemoryBank:
        pass
    
    print("Step 6: Initialize orchestrator with real config")
    model = MockModel()
    tokenizer = MockTokenizer()
    memory_bank = MockMemoryBank()
    device = torch.device('cpu')
    
    orchestrator = UnifiedRefinerOrchestrator(
        model=model,
        tokenizer=tokenizer,
        memory_bank=memory_bank,
        config=config,
        device=device
    )
    
    print("SUCCESS: Orchestrator created without error")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()