#!/usr/bin/env python3
"""
Debug script to find the torch.empty() issue in CognateTrainingPipeline
"""

import torch
import sys
import traceback
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.resolve()
packages_path = script_dir / "packages"
cognate_path = script_dir / "core" / "agent_forge" / "phases" / "cognate_pretrain"
sys.path.insert(0, str(packages_path))
sys.path.insert(0, str(cognate_path))

try:
    from agent_forge.models.cognate.training_pipeline import CognateTrainingPipeline, TrainingConfig
    from full_cognate_25m import Enhanced25MCognate, create_three_25m_models
    from refiner_core import CognateConfig
    
    print("Imports successful")
    
    # Create minimal config  
    config = TrainingConfig(
        vocab_size=32000,
        batch_size=2,
        max_steps=10,
        t_max_train=2,
        t_min_train=1
    )
    
    print("Config created")
    
    # Create minimal model
    models = create_three_25m_models()
    model = models[0]
    
    print("Model created")
    
    # Create minimal tokenizer
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 32000
            self.pad_token_id = 0
            self.bos_token_id = 1
            self.eos_token_id = 2

    tokenizer = MockTokenizer()
    
    # Create training pipeline
    pipeline = CognateTrainingPipeline(config, model.cognate_core, tokenizer)
    
    print("Pipeline created")
    
    # Create minimal batch
    batch = {
        "input_ids": torch.randint(0, 32000, (2, 10), dtype=torch.long),
        "labels": torch.randint(0, 32000, (2, 10), dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
        "seq_type": ["short"] * 2,
        "requires_memory": [False] * 2,
    }
    
    print("Batch created")
    
    # Create minimal optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    
    print("Optimizer created")
    
    # Try the train_step that's failing
    print("Attempting train_step...")
    result = pipeline.train_step(batch, optimizer, scheduler)
    
    print(f"Train step successful: {result}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()