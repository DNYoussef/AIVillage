#!/usr/bin/env python3
"""Simple script to update config/ references to configs/"""

import os
import re
from pathlib import Path

def update_file(file_path, replacements):
    """Update a file with replacements"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all config/ references"""
    
    # Define replacements
    replacements = {
        'config/hyperag_mcp.yaml': 'configs/hyperag_mcp.yaml',
        'config/compression.yaml': 'configs/compression.yaml', 
        'config/retrieval.yaml': 'configs/retrieval.yaml',
        'config/gdc_rules.yaml': 'configs/gdc_rules.yaml',
        'config/scanner_config.json': 'configs/scanner_config.yaml',
        '"config/': '"configs/',
        "'config/": "'configs/",
        '`config/': '`configs/',
        './config/': './configs/',
        'Check config/': 'Check configs/'
    }
    
    # Files to update
    files_to_update = [
        '.env.mcp',
        'mcp_servers/hyperag/server.py',
        'scripts/hyperag_scan_gdc.py',
        'mcp_servers/hyperag/README.md',
        'agent_forge/self_evolution_engine.py',
        '.github/workflows/compression-tests.yml',
        '.github/PULL_REQUEST_TEMPLATE/compression.md',
        'docs/compression_guide.md',
        'STYLE_GUIDE.md',
        '.pre-commit-config.yaml',
        'jobs/README.md',
        'jobs/crontab_hyperag',
    ]
    
    # Docker files
    docker_files = [
        'deploy/docker/Dockerfile.agent-forge',
        'deploy/docker/Dockerfile.compression-service',
        'deploy/docker/Dockerfile.credits-api',
        'deploy/docker/Dockerfile.credits-worker',
        'deploy/docker/Dockerfile.evolution-engine',
        'deploy/docker/Dockerfile.gateway',
        'deploy/docker/Dockerfile.hyperag-mcp',
        'deploy/docker/Dockerfile.mesh-network',
        'deploy/docker/Dockerfile.twin',
    ]
    
    all_files = files_to_update + docker_files
    updated_count = 0
    
    for file_path in all_files:
        path = Path(file_path)
        if path.exists():
            if update_file(path, replacements):
                updated_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nUpdated {updated_count} files with config path references")

if __name__ == "__main__":
    main()