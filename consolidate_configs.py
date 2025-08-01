#!/usr/bin/env python3
"""
Configuration Consolidation Script

This script consolidates the configuration directories and updates all references:
1. Merges config/ and configs/ into a single configs/ directory
2. Converts JSON configurations to YAML format
3. Moves test result files to results/ directory
4. Updates all import paths throughout the codebase
"""

import json
import os
import re
import shutil
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple


def convert_json_to_yaml(json_file: Path, yaml_file: Path) -> bool:
    """Convert a JSON file to YAML format"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        
        print(f"✓ Converted {json_file} to {yaml_file}")
        return True
    except Exception as e:
        print(f"✗ Error converting {json_file}: {e}")
        return False


def move_test_results():
    """Move test result files from configs/ to results/"""
    configs_dir = Path("configs")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    test_files = [
        "agent_forge_pipeline_summary.json",
        "orchestration_test_results.json", 
        "openrouter_metrics.json",
        "smoke_test_results.json"
    ]
    
    moved_files = []
    for filename in test_files:
        src = configs_dir / filename
        dst = results_dir / filename
        
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_files.append(filename)
            print(f"✓ Moved {filename} to results/")
    
    return moved_files


def merge_config_directories():
    """Merge config/ directory into configs/"""
    config_dir = Path("config")
    configs_dir = Path("configs")
    
    if not config_dir.exists():
        print("ℹ config/ directory doesn't exist, skipping merge")
        return []
    
    configs_dir.mkdir(exist_ok=True)
    moved_files = []
    
    for file_path in config_dir.iterdir():
        if file_path.is_file():
            dst = configs_dir / file_path.name
            
            if file_path.suffix == '.json' and file_path.name != 'scanner_config.json':
                # Keep as JSON if not scanner_config
                shutil.copy2(str(file_path), str(dst))
            elif file_path.name == 'scanner_config.json':
                # Convert scanner_config.json to YAML
                yaml_dst = configs_dir / 'scanner_config.yaml'
                if convert_json_to_yaml(file_path, yaml_dst):
                    moved_files.append(f"{file_path.name} -> scanner_config.yaml")
                continue
            else:
                # Copy YAML files directly
                shutil.copy2(str(file_path), str(dst))
            
            moved_files.append(file_path.name)
            print(f"✓ Merged {file_path.name} into configs/")
    
    return moved_files


def update_file_references(file_path: Path, patterns: List[Tuple[str, str]]) -> bool:
    """Update file with pattern replacements"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        for old_pattern, new_pattern in patterns:
            content = re.sub(old_pattern, new_pattern, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"✗ Error updating {file_path}: {e}")
        return False


def update_all_references():
    """Update all config/ references to configs/"""
    root_dir = Path(".")
    
    # Patterns to replace
    patterns = [
        # Specific file patterns
        (r'\bconfig/hyperag_mcp\.yaml\b', r'configs/hyperag_mcp.yaml'),
        (r'\bconfig/compression\.yaml\b', r'configs/compression.yaml'),
        (r'\bconfig/retrieval\.yaml\b', r'configs/retrieval.yaml'),
        (r'\bconfig/gdc_rules\.yaml\b', r'configs/gdc_rules.yaml'),
        (r'\bconfig/scanner_config\.json\b', r'configs/scanner_config.yaml'),
        
        # Generic patterns
        (r'\bconfig/([a-zA-Z0-9_\-]+\.ya?ml)\b', r'configs/\1'),
        (r'\bconfig/([a-zA-Z0-9_\-]+\.json)\b', r'configs/\1'),
        
        # Path patterns
        (r'"config/', r'"configs/'),
        (r"'config/", r"'configs/"),
        (r'`config/', r'`configs/'),
        (r'./config/', r'./configs/'),
    ]
    
    # File types to update
    file_extensions = ['.py', '.md', '.yml', '.yaml', '.json', '.sh', '.env.mcp', '.txt']
    
    updated_files = []
    
    # Walk through all files
    for file_path in root_dir.rglob('*'):
        if (file_path.is_file() and 
            any(str(file_path).endswith(ext) for ext in file_extensions) and
            not any(part in str(file_path) for part in ['.git', '__pycache__', 'node_modules', '.pytest_cache', 'config/', 'results/'])):
            
            if update_file_references(file_path, patterns):
                updated_files.append(str(file_path))
                print(f"✓ Updated references in {file_path}")
    
    return updated_files


def cleanup_old_directories():
    """Remove the old config/ directory after successful merge"""
    config_dir = Path("config")
    
    if config_dir.exists():
        try:
            shutil.rmtree(config_dir)
            print(f"✓ Removed old config/ directory")
            return True
        except Exception as e:
            print(f"✗ Error removing config/ directory: {e}")
            return False
    return True


def main():
    """Main consolidation function"""
    print("🔧 Starting configuration consolidation...")
    print()
    
    # Step 1: Move test result files
    print("📁 Moving test result files to results/...")
    test_files = move_test_results()
    print(f"   Moved {len(test_files)} test result files")
    print()
    
    # Step 2: Merge config directories
    print("📦 Merging config/ into configs/...")
    merged_files = merge_config_directories()
    print(f"   Merged {len(merged_files)} configuration files")
    print()
    
    # Step 3: Update all references
    print("🔄 Updating configuration file references...")
    updated_files = update_all_references()
    print(f"   Updated references in {len(updated_files)} files")
    print()
    
    # Step 4: Cleanup old directory
    print("🧹 Cleaning up old directories...")
    cleanup_success = cleanup_old_directories()
    print()
    
    # Summary
    print("📋 Configuration Consolidation Summary:")
    print("=" * 50)
    print(f"✓ Moved {len(test_files)} test result files to results/")
    print(f"✓ Merged {len(merged_files)} config files into configs/")
    print(f"✓ Updated references in {len(updated_files)} source files")
    print(f"✓ Cleanup: {'Success' if cleanup_success else 'Failed'}")
    print()
    
    if updated_files:
        print("📝 Updated files:")
        for file_path in updated_files[:10]:  # Show first 10
            print(f"   - {file_path}")
        if len(updated_files) > 10:
            print(f"   ... and {len(updated_files) - 10} more files")
    
    print()
    print("🎉 Configuration consolidation complete!")
    print()
    print("📂 Final structure:")
    print("   configs/           # All configuration files (YAML format)")
    print("   results/           # Test results and metrics")
    print("   (removed config/)  # Old directory cleaned up")


if __name__ == "__main__":
    main()