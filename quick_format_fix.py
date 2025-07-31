#!/usr/bin/env python3
"""
Quick formatting fixes for Python files
Runs black and isort on all Python files in the project.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    project_root = Path(__file__).parent
    
    # Get all Python files
    python_files = []
    for pattern in ["agent_forge/**/*.py", "mcp_servers/**/*.py", "tests/**/*.py", "production/**/*.py", "scripts/**/*.py", "benchmarks/**/*.py"]:
        python_files.extend(project_root.glob(pattern))
    
    # Filter out files to exclude
    exclude_patterns = ["new_env", "__pycache__", ".git", ".cleanup_backups", ".test_repair_backup"]
    python_files = [f for f in python_files if not any(pattern in str(f) for pattern in exclude_patterns)]
    
    if not python_files:
        print("No Python files found to format")
        return 0
    
    print(f"Found {len(python_files)} Python files to format")
    file_paths = [str(f) for f in python_files]
    
    # Install packages if needed
    for package in ["black", "isort"]:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    
    # Run black formatting
    print("Running black formatter...")
    success, stdout, stderr = run_command([sys.executable, "-m", "black"] + file_paths)
    if success:
        print("✓ Black formatting completed successfully")
    else:
        print(f"✗ Black formatting failed: {stderr}")
        return 1
    
    # Run isort import organization
    print("Running isort import organizer...")
    success, stdout, stderr = run_command([sys.executable, "-m", "isort"] + file_paths)
    if success:
        print("✓ Import organization completed successfully")
    else:
        print(f"✗ Import organization failed: {stderr}")
        return 1
    
    print("\n✓ All formatting fixes applied successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())