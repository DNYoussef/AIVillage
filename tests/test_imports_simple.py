#!/usr/bin/env python3
import sys
import os

# Add the necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core', 'agent-forge', 'phases', 'cognate_pretrain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core', 'agent-forge', 'models', 'cognate', 'training'))

try:
    import enhanced_trainer
    print("SUCCESS: enhanced_trainer imports successfully")
except Exception as e:
    print(f"FAILED: enhanced_trainer import failed: {e}")

try:
    from real_pretraining_pipeline import REAL_IMPORTS
    print(f"SUCCESS: real_pretraining_pipeline REAL_IMPORTS = {REAL_IMPORTS}")
except Exception as e:
    print(f"FAILED: real_pretraining_pipeline import failed: {e}")