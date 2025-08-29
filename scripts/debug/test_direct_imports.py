#!/usr/bin/env python3
"""
Test script to validate Agent Forge imports using direct file paths
"""

import sys
import os
import importlib.util
from pathlib import Path

def import_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add to sys.modules for relative imports
    spec.loader.exec_module(module)
    return module

def test_direct_imports():
    """Test importing Agent Forge components directly."""
    try:
        print("Testing Direct Agent Forge Imports...")
        print("-" * 50)
        
        # Get project root directory (go up 2 levels from scripts/debug/)
        project_root = Path(__file__).parent.parent.parent
        
        # First import the core phase controller
        phase_controller_path = project_root / "core/agent-forge/core/phase_controller.py"
        phase_controller = import_from_path("phase_controller", str(phase_controller_path))
        print("+ PhaseController module loaded")
        
        # Now try to import each phase file individually 
        phases_dir = project_root / "core/agent-forge/phases"
        
        working_phases = []
        failing_phases = []
        
        phase_files = [
            "evomerge.py",
            "quietstar.py", 
            "bitnet_compression.py",
            "forge_training.py",
            "tool_persona_baking.py",
            "adas.py",
            "final_compression.py",
        ]
        
        for phase_file in phase_files:
            try:
                phase_path = phases_dir / phase_file
                if not phase_path.exists():
                    failing_phases.append(f"{phase_file} -> file not found")
                    continue
                    
                phase_name = phase_file.replace('.py', '')
                phase_module = import_from_path(phase_name, str(phase_path))
                
                # Look for phase classes
                phase_classes = []
                for attr_name in dir(phase_module):
                    attr = getattr(phase_module, attr_name)
                    if isinstance(attr, type) and attr_name.endswith('Phase'):
                        phase_classes.append(attr_name)
                
                if phase_classes:
                    working_phases.append(f"{phase_file} -> {', '.join(phase_classes)}")
                    print(f"+ {phase_file}: Found {', '.join(phase_classes)}")
                else:
                    failing_phases.append(f"{phase_file} -> no phase classes found")
                    print(f"- {phase_file}: No phase classes found")
                    
            except Exception as e:
                failing_phases.append(f"{phase_file} -> {e}")
                print(f"X {phase_file}: {e}")
        
        print("-" * 50)
        print(f"Working phases: {len(working_phases)}")
        print(f"Failing phases: {len(failing_phases)}")
        
        if working_phases:
            print(f"\n+ Found {len(working_phases)} working phase files")
            for phase in working_phases:
                print(f"  - {phase}")
            return True
        else:
            print("\nX No working phases found")
            return False
            
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_imports()
    exit(0 if success else 1)