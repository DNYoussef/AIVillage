#!/usr/bin/env python3
"""
EvoMerge 50-Generation Runner - Compatibility Wrapper
DEPRECATED: This file has moved to packages/agent_forge/experiments/run_evomerge_50gen.py
This wrapper maintains backward compatibility and will be removed on September 20, 2025.
"""

import os
from pathlib import Path
import sys
import warnings

# Issue deprecation warning
warnings.warn(
    "This script has moved to packages/agent_forge/experiments/run_evomerge_50gen.py. "
    "Please update your scripts to use the new location. "
    "This compatibility wrapper will be removed on September 20, 2025.",
    DeprecationWarning,
    stacklevel=2,
)

# Add packages directory to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

# Import and execute the actual script
try:
    # Change to the experiments directory
    original_cwd = os.getcwd()
    experiments_dir = Path(__file__).parent / "packages" / "agent_forge" / "experiments"
    os.chdir(experiments_dir)

    # Import and run the main function
    from run_evomerge_50gen import main

    print("[DEPRECATED] Using compatibility wrapper. Please run:")
    print("  cd packages/agent_forge/experiments && python run_evomerge_50gen.py")
    print()

    exit_code = main()

finally:
    # Restore original working directory
    os.chdir(original_cwd)

sys.exit(exit_code)
