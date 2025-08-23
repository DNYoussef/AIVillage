#!/usr/bin/env python3
"""
Import Path Migration Script - Agent 6 Final Cleanup
Systematically updates all deprecated import paths to consolidated locations
"""

import os
import re
from pathlib import Path

# Import path mappings: OLD -> NEW
IMPORT_MIGRATIONS = {
    # Agent imports
    r'from packages\.agents\.core\.base import': 'from agents.core.base import',
    r'from packages\.core\.legacy\.error_handling import': 'from core.legacy.error_handling import',
    
    # RAG imports  
    r'from core\.rag\.hyper_rag import': 'from rag.core.hyper_rag import',
    r'from packages\.rag\.core\.hyper_rag import': 'from rag.core.hyper_rag import',
    
    # Gateway imports
    r'from core\.gateway\.server import': 'from gateway.server import',
    
    # P2P imports
    r'from core\.p2p\.mesh_protocol import': 'from p2p.mesh_protocol import',
}

def migrate_imports_in_file(file_path: Path) -> bool:
    """
    Migrate import statements in a single file.
    Returns True if any changes were made.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        for old_pattern, new_import in IMPORT_MIGRATIONS.items():
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_import, content)
                changes_made = True
                print(f"  ✅ Updated: {old_pattern} -> {new_import}")
        
        if changes_made:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False

def find_python_files(root_dir: Path) -> list[Path]:
    """Find all Python files that might need import updates."""
    python_files = []
    
    for pattern in ['**/*.py']:
        python_files.extend(root_dir.glob(pattern))
    
    # Filter out backup files and __pycache__
    return [f for f in python_files if not f.name.endswith('.backup') and '__pycache__' not in str(f)]

def main():
    print("🔧 IMPORT PATH MIGRATION - Agent 6 Final Cleanup")
    print("=" * 60)
    
    root_dir = Path('.')
    python_files = find_python_files(root_dir)
    
    print(f"📁 Found {len(python_files)} Python files to check")
    print()
    
    files_updated = 0
    total_changes = 0
    
    for file_path in python_files:
        if migrate_imports_in_file(file_path):
            print(f"📝 Updated: {file_path}")
            files_updated += 1
            total_changes += 1
    
    print()
    print("📊 MIGRATION SUMMARY")
    print("-" * 30)
    print(f"Files processed: {len(python_files)}")
    print(f"Files updated: {files_updated}")
    print(f"Total changes: {total_changes}")
    
    if files_updated > 0:
        print()
        print("✅ Import migration completed successfully!")
        print("💾 Backup files created with .backup extension")
        print("🔄 You can rollback by restoring .backup files if needed")
    else:
        print("ℹ️ No import paths needed updating")

if __name__ == "__main__":
    main()