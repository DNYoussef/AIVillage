#!/usr/bin/env python3
"""
Fixed Standalone Validation for Windows
Handles Unicode encoding issues and provides safe test execution.
"""

import os
import sys
from pathlib import Path

# Fix Windows encoding issues
if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"


def analyze_file_content(file_path, expected_patterns):
    """Analyze a file for expected patterns."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        results = {}
        for pattern_name, pattern in expected_patterns.items():
            results[pattern_name] = pattern.lower() in content.lower()

        return True, results
    except Exception as e:
        return False, {"error": str(e)}


def validate_file_structure():
    """Validate the file structure is complete."""
    project_root = Path(__file__).parent.parent.parent
    # Try both naming conventions
    cognate_pretrain_dir = project_root / "core" / "agent_forge" / "phases" / "cognate_pretrain"
    if not cognate_pretrain_dir.exists():
        cognate_pretrain_dir = project_root / "core" / "agent_forge" / "phases" / "cognate-pretrain"

    print("=" * 60)
    print("COGNATE 25M SYSTEM - STANDALONE VALIDATION (FIXED)")
    print("=" * 60)

    # Check directory exists
    print("1. DIRECTORY STRUCTURE")
    print("-" * 30)
    print(f"Cognate pretrain directory: {cognate_pretrain_dir}")

    if not cognate_pretrain_dir.exists():
        print("CRITICAL ERROR: Cognate pretrain directory not found!")
        return False
    print("OK Directory exists")

    # Check required files
    required_files = {
        "__init__.py": "Package initialization",
        "model_factory.py": "Main entry point",
        "cognate_creator.py": "Core model creation",
        "pretrain_pipeline.py": "Optional pre-training",
        "phase_integration.py": "Agent Forge integration",
        "README.md": "Documentation",
    }

    all_files_present = True
    for filename, description in required_files.items():
        file_path = cognate_pretrain_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"OK {filename} ({file_size:,} bytes) - {description}")
        else:
            print(f"MISSING {filename} - {description}")
            all_files_present = False

    return all_files_present


def validate_core_functionality():
    """Validate core functionality by analyzing code."""
    project_root = Path(__file__).parent.parent.parent
    # Try both naming conventions
    cognate_pretrain_dir = project_root / "core" / "agent_forge" / "phases" / "cognate_pretrain"
    if not cognate_pretrain_dir.exists():
        cognate_pretrain_dir = project_root / "core" / "agent_forge" / "phases" / "cognate-pretrain"

    print("\n2. CORE FUNCTIONALITY ANALYSIS")
    print("-" * 30)

    # Analyze model_factory.py
    factory_file = cognate_pretrain_dir / "model_factory.py"
    if factory_file.exists():
        success, results = analyze_file_content(
            factory_file,
            {
                "create_three_cognate_models": "create_three_cognate_models",
                "validate_cognate_models": "validate_cognate_models",
                "25m_parameters": "25_000_000",
                "evomerge_ready": "ready_for_evomerge",
            },
        )

        if success:
            print("OK model_factory.py analysis:")
            for pattern, found in results.items():
                status = "OK" if found else "MISSING"
                print(f"  {status} {pattern}")
        else:
            print(f"FAILED model_factory.py analysis: {results.get('error')}")
    else:
        print("MISSING model_factory.py not found")

    # Analyze cognate_creator.py
    creator_file = cognate_pretrain_dir / "cognate_creator.py"
    if creator_file.exists():
        success, results = analyze_file_content(
            creator_file,
            {
                "CognateModelCreator": "CognateModelCreator",
                "CognateCreatorConfig": "CognateCreatorConfig",
                "25m_params": "25_000_000",
                "three_models": "create_three_models",
            },
        )

        if success:
            print("\nOK cognate_creator.py analysis:")
            for pattern, found in results.items():
                status = "OK" if found else "MISSING"
                print(f"  {status} {pattern}")
        else:
            print(f"\nFAILED cognate_creator.py analysis: {results.get('error')}")
    else:
        print("\nMISSING cognate_creator.py not found")


def validate_integration():
    """Validate integration with Agent Forge pipeline."""
    print("\n3. AGENT FORGE INTEGRATION")
    print("-" * 30)

    project_root = Path(__file__).parent.parent.parent

    # Check for EvoMerge integration
    evomerge_file = project_root / "core" / "agent_forge" / "phases" / "evomerge.py"
    if evomerge_file.exists():
        print("OK EvoMerge phase file exists")

        # Check if it references Cognate
        success, results = analyze_file_content(
            evomerge_file, {"cognate": "cognate", "merge": "merge", "evolution": "evolution"}
        )

        if success:
            for pattern, found in results.items():
                status = "OK" if found else "MISSING"
                print(f"  {status} EvoMerge {pattern} references")
    else:
        print("MISSING EvoMerge phase file")

    # Check for existing Cognate models
    cognate_output_dirs = [
        project_root / "core" / "agent_forge" / "phases" / "cognate-pretrain" / "models",
        project_root / "core" / "agent_forge" / "phases" / "trained_25m_models",
    ]

    for output_dir in cognate_output_dirs:
        if output_dir.exists():
            models = list(output_dir.iterdir())
            print(f"OK Found {len(models)} items in {output_dir.name}")
        else:
            print(f"MISSING {output_dir.name} directory")


def main():
    """Main validation function with error handling."""
    try:
        structure_ok = validate_file_structure()
        validate_core_functionality()
        validate_integration()

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if structure_ok:
            print("STATUS: Core file structure is present")
            print("VERDICT: Cognate system appears to be properly organized")
            print("READY: Can proceed with testing and EvoMerge integration")
        else:
            print("STATUS: File structure has issues")
            print("VERDICT: Cognate system needs attention")
            print("ACTION: Fix missing files before proceeding")

        return structure_ok

    except Exception as e:
        print(f"CRITICAL ERROR: Validation failed - {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
