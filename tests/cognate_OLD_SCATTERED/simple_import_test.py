#!/usr/bin/env python3
"""
Simple import test to debug the module structure
"""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Also need to add the core directory specifically
core_path = project_root / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path))

print("Python path:")
for path in sys.path[:5]:  # Show first 5 paths
    print(f"  {path}")

print(f"\nProject root: {project_root}")
print(f"Core path: {core_path}")
print(f"Core exists: {core_path.exists()}")

# Check what's actually in the core directory
agent_forge_path = core_path / "agent-forge"
print(f"Agent-forge path: {agent_forge_path}")
print(f"Agent-forge exists: {agent_forge_path.exists()}")

if agent_forge_path.exists():
    phases_path = agent_forge_path / "phases"
    print(f"Phases path: {phases_path}")
    print(f"Phases exists: {phases_path.exists()}")
    
    if phases_path.exists():
        cognate_pretrain_path = phases_path / "cognate-pretrain"
        print(f"Cognate-pretrain path: {cognate_pretrain_path}")
        print(f"Cognate-pretrain exists: {cognate_pretrain_path.exists()}")

# Try different import strategies
print("\nTesting imports:")

# Strategy 1: Direct module import
try:
    print("✅ agent_forge import worked")
except ImportError as e:
    print(f"❌ agent_forge import failed: {e}")

# Strategy 2: Try with hyphen replacement
try:
    sys.path.insert(0, str(agent_forge_path.parent))
    print("✅ agent_forge import with path adjustment worked")
except ImportError as e:
    print(f"❌ agent_forge import with path adjustment failed: {e}")

# Strategy 3: Try direct file import
try:
    cognate_pretrain_path = agent_forge_path / "phases" / "cognate-pretrain"
    if cognate_pretrain_path.exists():
        sys.path.insert(0, str(cognate_pretrain_path.parent))
        
        # Try to manually load the module
        import importlib.util
        init_file = cognate_pretrain_path / "__init__.py"
        if init_file.exists():
            spec = importlib.util.spec_from_file_location("cognate_pretrain", init_file)
            if spec and spec.loader:
                cognate_pretrain = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cognate_pretrain)
                print("✅ Direct cognate-pretrain module import worked")
                
                if hasattr(cognate_pretrain, 'create_three_cognate_models'):
                    print("✅ create_three_cognate_models function found")
                else:
                    print("❌ create_three_cognate_models function not found")
        else:
            print(f"❌ __init__.py not found in {cognate_pretrain_path}")
    else:
        print(f"❌ cognate-pretrain directory not found: {cognate_pretrain_path}")
        
except Exception as e:
    print(f"❌ Direct module import failed: {e}")

print("\nDone with import testing.")