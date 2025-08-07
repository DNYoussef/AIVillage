#!/usr/bin/env python3
"""Execute the documentation cleanup step by step."""

import os
from pathlib import Path
import subprocess
import sys

# Change to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

# Execute the main cleanup script
try:
    result = subprocess.run(
        [sys.executable, "cleanup_documentation.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\nReturn code: {result.returncode}")

except Exception as e:
    print(f"Error executing cleanup: {e}")

    # If subprocess fails, try importing and running directly
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from cleanup_documentation import DocumentationCleanup

        print("Running cleanup directly...")
        cleanup = DocumentationCleanup()
        cleanup.run_cleanup()

    except Exception as e2:
        print(f"Direct execution also failed: {e2}")
