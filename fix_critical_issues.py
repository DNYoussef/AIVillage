#!/usr/bin/env python3
"""
Fix critical linting issues in Agent Forge codebase.
Focuses on import organization, formatting, and common issues.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

def read_file(file_path: Path) -> str:
    """Read file content safely."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(file_path: Path, content: str) -> None:
    """Write file content safely."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def organize_imports(content: str) -> str:
    """Organize imports according to PEP8 standards."""
    lines = content.split('\n')
    
    # Find import section
    import_start = None
    import_end = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')) and import_start is None:
            import_start = i
        elif import_start is not None and not line.strip().startswith(('import ', 'from ')) and line.strip() and not line.strip().startswith('#'):
            import_end = i
            break
    
    if import_start is None:
        return content
    
    if import_end is None:
        import_end = len(lines)
    
    # Extract imports
    import_lines = lines[import_start:import_end]
    
    # Separate different types of imports
    standard_imports = []
    third_party_imports = []
    local_imports = []
    
    for line in import_lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('from .') or line.startswith('from agent_forge') or line.startswith('from mcp_servers'):
            local_imports.append(line)
        elif any(lib in line for lib in ['torch', 'transformers', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'jinja2']):
            third_party_imports.append(line)
        else:
            standard_imports.append(line)
    
    # Sort each group
    standard_imports.sort()
    third_party_imports.sort()
    local_imports.sort()
    
    # Rebuild imports section
    new_imports = []
    if standard_imports:
        new_imports.extend(standard_imports)
        new_imports.append('')
    if third_party_imports:
        new_imports.extend(third_party_imports)
        new_imports.append('')
    if local_imports:
        new_imports.extend(local_imports)
        new_imports.append('')
    
    # Rebuild file
    new_lines = lines[:import_start] + new_imports + lines[import_end:]
    return '\n'.join(new_lines)

def fix_long_lines(content: str) -> str:
    """Fix lines that are too long."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > 88:  # Black's default line length
            # Simple fixes for common cases
            if ' and ' in line and len(line) > 100:
                # Break long boolean expressions
                parts = line.split(' and ')
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    fixed_line = parts[0] + ' and \\\n' + ' ' * (indent + 4) + ' and '.join(parts[1:])
                    fixed_lines.append(fixed_line)
                    continue
            
            # Don't modify string literals or complex expressions
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_print_statements(content: str) -> str:
    """Replace print statements with logging in non-CLI contexts."""
    lines = content.split('\n')
    fixed_lines = []
    
    # Check if this is a CLI script
    is_cli = any('if __name__ == "__main__"' in line for line in lines)
    
    for i, line in enumerate(lines):
        if line.strip().startswith('print(') and not is_cli:
            # In the main execution context, keep prints
            in_main_block = False
            for j in range(max(0, i-10), min(len(lines), i+5)):
                if 'if __name__ == "__main__"' in lines[j]:
                    in_main_block = True
                    break
            
            if not in_main_block:
                # Replace print with logger
                indent = len(line) - len(line.lstrip())
                content_match = re.search(r'print\((.*)\)', line.strip())
                if content_match:
                    print_content = content_match.group(1)
                    # Simple replacement - more complex logic would be needed for formatting
                    if print_content.startswith('f"') or print_content.startswith("f'"):
                        new_line = ' ' * indent + f'logger.info({print_content})'
                    else:
                        new_line = ' ' * indent + f'logger.info({print_content})'
                    fixed_lines.append(new_line)
                    continue
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def add_missing_imports(content: str) -> str:
    """Add missing imports that are commonly needed."""
    lines = content.split('\n')
    
    # Check if logging is used but not imported
    uses_logger = any('logger.' in line for line in lines)
    has_logging_import = any('import logging' in line for line in lines)
    
    if uses_logger and not has_logging_import:
        # Add logging import
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('from __future__'):
                lines.insert(i, 'import logging')
                break
    
    return '\n'.join(lines)

def fix_common_issues(content: str) -> str:
    """Fix common linting issues."""
    # Remove trailing whitespace
    lines = [line.rstrip() for line in content.split('\n')]
    
    # Ensure file ends with newline
    if lines and lines[-1]:
        lines.append('')
    
    # Remove multiple consecutive blank lines
    fixed_lines = []
    blank_count = 0
    
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:  # Allow max 2 consecutive blank lines
                fixed_lines.append(line)
        else:
            blank_count = 0
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def process_file(file_path: Path) -> bool:
    """Process a single Python file for quality issues."""
    try:
        print(f"Processing {file_path.relative_to(Path.cwd())}")
        
        content = read_file(file_path)
        original_content = content
        
        # Apply fixes
        content = organize_imports(content)
        content = fix_print_statements(content)
        content = add_missing_imports(content)
        content = fix_common_issues(content)
        
        # Only write if changed
        if content != original_content:
            write_file(file_path, content)
            print(f"  ✓ Fixed issues in {file_path.name}")
            return True
        else:
            print(f"  - No issues found in {file_path.name}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix critical issues."""
    project_root = Path(__file__).parent
    
    # Target directories
    target_dirs = [
        "agent_forge",
        "mcp_servers", 
        "production",
        "tests"
    ]
    
    files_processed = 0
    files_fixed = 0
    
    for target_dir in target_dirs:
        dir_path = project_root / target_dir
        if not dir_path.exists():
            continue
            
        print(f"\nProcessing {target_dir}/")
        
        # Get Python files
        py_files = list(dir_path.rglob("*.py"))
        
        # Filter out problematic paths
        py_files = [f for f in py_files if not any(exclude in str(f) for exclude in [
            "new_env", "__pycache__", ".git", ".cleanup_backups", ".test_repair_backup"
        ])]
        
        for py_file in py_files:
            files_processed += 1
            if process_file(py_file):
                files_fixed += 1
    
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"Files processed: {files_processed}")
    print(f"Files fixed: {files_fixed}")
    
    if files_fixed > 0:
        print(f"\n✓ Fixed critical issues in {files_fixed} files")
        
        # Run black formatting
        print("\nRunning black formatter...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "black", 
                "agent_forge", "mcp_servers", "production", "tests",
                "--exclude", "new_env|__pycache__|.git"
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print("✓ Black formatting completed")
            else:
                print(f"⚠ Black formatting had issues: {result.stderr}")
        except Exception as e:
            print(f"⚠ Could not run black: {e}")
        
    else:
        print("✓ No critical issues found")
    
    return 0 if files_fixed < 50 else 1  # Allow some issues

if __name__ == "__main__":
    sys.exit(main())