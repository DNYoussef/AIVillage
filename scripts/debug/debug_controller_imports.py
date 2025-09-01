#!/usr/bin/env python3
"""
Debug script to test the exact import logic from the controller
"""

import os
from pathlib import Path
import sys

# Exact same setup as the controller
project_root = Path(".")
os.environ["PYTHONPATH"] = str(project_root) + ";" + str(project_root / "core")

# Initialize global variables at module level (exact same as controller)
RealCognateTrainer = None
RealTrainingConfig = None
CognateModelCreator = None
CognateCreatorConfig = None
create_three_cognate_models = None
REAL_TRAINING_AVAILABLE = False  # Initialize here to avoid NameError

print("DEBUG: Real Cognate pretraining system initializing...")
print(f"DEBUG: Project root: {project_root}")
print(f"DEBUG: PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# Add cognate_pretrain directory to Python path (exact same as controller)
cognate_pretrain_dir = project_root / "core" / "agent_forge" / "phases" / "cognate_pretrain"
if str(cognate_pretrain_dir) not in sys.path:
    sys.path.insert(0, str(cognate_pretrain_dir))

print(f"DEBUG: Added to path: {cognate_pretrain_dir}")
print(f"DEBUG: Directory exists: {cognate_pretrain_dir.exists()}")

try:
    # EXACT same import block as controller
    from full_pretraining_pipeline import FullCognateTrainer, FullPretrainingConfig

    # Assign the real training components (exact same as controller)
    RealCognateTrainer = FullCognateTrainer
    RealTrainingConfig = FullPretrainingConfig
    REAL_TRAINING_AVAILABLE = True

    print("[SUCCESS] PRODUCTION REAL TRAINING COMPONENTS LOADED SUCCESSFULLY")
    print("  - FullCognateTrainer: Available")
    print("  - FullPretrainingConfig: Available")
    print("  - RealCognateDataset: Available")
    print("  - Enhanced25MCognate: Available")
    print("  - Production real training with datasets: ENABLED")

except ImportError as e:
    print(f"[ERROR] Real training import failed: {e}")
    print(f"   Trying to import from: {cognate_pretrain_dir}")
    # Check if files exist for debugging (exact same as controller)
    real_pipeline_file = cognate_pretrain_dir / "real_pretraining_pipeline.py"
    full_cognate_file = cognate_pretrain_dir / "full_cognate_25m.py"
    print(f"   real_pretraining_pipeline.py exists: {real_pipeline_file.exists()}")
    print(f"   full_cognate_25m.py exists: {full_cognate_file.exists()}")

    import traceback

    traceback.print_exc()

# Test the exact same status logic as controller
print("\n=== STATUS CHECK (same as controller) ===")
print(f"REAL_TRAINING_AVAILABLE: {REAL_TRAINING_AVAILABLE}")
print(f"RealCognateTrainer: {bool(RealCognateTrainer)} ({type(RealCognateTrainer)})")
print(f"RealTrainingConfig: {bool(RealTrainingConfig)} ({type(RealTrainingConfig)})")
print(f"Enhanced25MCognate in globals(): {'Enhanced25MCognate' in globals()}")

training_system_status = {
    "real_cognate_trainer": bool(RealCognateTrainer) if REAL_TRAINING_AVAILABLE else False,
    "real_training_config": bool(RealTrainingConfig) if REAL_TRAINING_AVAILABLE else False,
    "enhanced_25m_cognate": "Enhanced25MCognate" in globals() if REAL_TRAINING_AVAILABLE else False,
    "mode": "REAL_TRAINING" if REAL_TRAINING_AVAILABLE else "FALLBACK_MODE",
}

print(f"Training system status: {training_system_status}")
