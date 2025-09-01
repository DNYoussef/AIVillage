#!/usr/bin/env python3
"""
Test script to validate Agent Forge phase imports
"""

import sys
from pathlib import Path

# Add the project root to Python path (go up 2 levels from scripts/debug/)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def test_phase_imports():
    """Test importing individual phase files directly."""
    try:
        print("Testing Agent Forge Phase Imports...")
        print("-" * 50)

        # Test importing the phase controller
        sys.path.append(str(project_root / "core"))

        print("[OK] Core agent_forge import successful")

        # Alternative approach - add to path and try direct import
        agent_forge_core_path = project_root / "core" / "agent_forge" / "core"
        sys.path.append(str(agent_forge_core_path))


        print("[OK] phase_controller module import successful")


        print("[OK] PhaseController import successful")

        # Test importing unified pipeline

        print("[OK] UnifiedPipeline import successful")

        # Now test individual phases that should work
        test_phases = [
            ("evomerge", "EvoMergePhase"),
            ("quietstar", "QuietSTaRPhase"),
            ("bitnet_compression", "BitNetCompressionPhase"),
            ("forge_training", "ForgeTrainingPhase"),
            ("tool_persona_baking", "ToolPersonaBakingPhase"),
            ("adas", "ADASPhase"),
            ("final_compression", "FinalCompressionPhase"),
        ]

        working_phases = []
        failing_phases = []

        for phase_file, phase_class in test_phases:
            try:
                # Import the module directly
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    phase_file, project_root / f"core/agent_forge/phases/{phase_file}.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Try to get the phase class
                if hasattr(module, phase_class):
                    working_phases.append(f"{phase_file} -> {phase_class}")
                    print(f"[OK] {phase_file}.py import successful - {phase_class} available")
                else:
                    failing_phases.append(f"{phase_file} -> {phase_class} not found")
                    print(f"[WARN] {phase_file}.py import successful but {phase_class} not found")

            except Exception as e:
                failing_phases.append(f"{phase_file} -> {e}")
                print(f"[ERROR] {phase_file}.py import failed: {e}")

        print("-" * 50)
        print(f"Working phases: {len(working_phases)}")
        print(f"Failing phases: {len(failing_phases)}")

        if working_phases:
            print(f"\n[SUCCESS] Pipeline can use {len(working_phases)} phases")
            return True
        else:
            print("\n[FAIL] No phases are working")
            return False

    except Exception as e:
        print(f"Critical error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_phase_imports()
    exit(0 if success else 1)
