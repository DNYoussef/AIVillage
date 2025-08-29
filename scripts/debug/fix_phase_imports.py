#!/usr/bin/env python3
"""
Script to fix imports in all Agent Forge phase files
"""

from pathlib import Path

# Fallback import code to add to each phase file
FALLBACK_IMPORT = '''# Try to import PhaseController, with fallback for direct imports
try:
    from ..core.phase_controller import PhaseController, PhaseResult
except (ImportError, ValueError):
    # Fallback for direct imports - create minimal base classes
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any
    import torch.nn as nn
    
    @dataclass
    class PhaseResult:
        success: bool
        model: nn.Module
        phase_name: str = None
        metrics: dict = None
        duration_seconds: float = 0.0
        artifacts: dict = None
        config: dict = None
        error: str = None
        start_time: datetime = None
        end_time: datetime = None
        
        def __post_init__(self):
            if self.end_time is None:
                self.end_time = datetime.now()
            if self.start_time is None:
                self.start_time = self.end_time
    
    class PhaseController(ABC):
        def __init__(self, config: Any):
            self.config = config
            
        @abstractmethod
        async def run(self, model: nn.Module) -> PhaseResult:
            pass'''

def fix_phase_file(file_path):
    """Fix imports in a single phase file."""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it needs fixing
        comment_marker = "# Import will be done inside methods to avoid relative import issues"
        if comment_marker not in content:
            print(f"Skipping {file_path.name} - no import marker found")
            return False
            
        # Replace the comment with the fallback import
        old_import = comment_marker + "\n# from ..core.phase_controller import PhaseController, PhaseResult"
        new_content = content.replace(old_import, FALLBACK_IMPORT)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"Fixed imports in {file_path.name}")
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path.name}: {e}")
        return False

def main():
    """Fix all phase files."""
    phases_dir = Path("core/agent-forge/phases")
    
    phase_files = [
        "bitnet_compression.py",
        "forge_training.py", 
        "tool_persona_baking.py",
        "adas.py",
        "final_compression.py"
    ]
    
    print("Fixing Agent Forge phase imports...")
    print("-" * 40)
    
    fixed_count = 0
    for phase_file in phase_files:
        file_path = phases_dir / phase_file
        if file_path.exists():
            if fix_phase_file(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {phase_file}")
    
    print("-" * 40)
    print(f"Fixed {fixed_count} phase files")

if __name__ == "__main__":
    main()