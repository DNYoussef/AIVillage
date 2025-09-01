#!/usr/bin/env python3
"""
Debug script to isolate the training pipeline tensor creation issue
"""

import torch
import sys
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.resolve()
packages_path = script_dir / "packages"
sys.path.insert(0, str(packages_path))

try:
    from agent_forge.models.cognate.training_pipeline import CognateDataset, CognateTrainingPipeline, TrainingConfig
    
    # Create a minimal test configuration
    config = TrainingConfig(
        vocab_size=32000,
        batch_size=2,
        max_steps=10
    )
    
    # Create synthetic data
    dataset = CognateDataset(config, data_files=[], tokenizer=None)
    
    # Try to create a model
    print("Config created successfully")
    print(f"Config vocab_size: {getattr(config, 'vocab_size', 'NOT_FOUND')}")
    
    # Try tensor creation with explicit parameters
    test_tensor = torch.randint(0, 32000, (2, 256), dtype=torch.long)
    print(f"Test tensor created: {test_tensor.shape}")
    
    print("Testing completed successfully")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc()