#!/usr/bin/env python3
"""
Batch Script to Fix Agent Forge Path References

This script converts all path references from "agent_forge" (hyphenated)
to "agent_forge" (underscored) to make the package name Python-compliant.
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_python_files(root_dir: str) -> List[Path]:
    """Find all Python files in the directory tree."""
    python_files = []
    root_path = Path(root_dir)

    for file_path in root_path.rglob("*.py"):
        # Skip files in .git, __pycache__, and other irrelevant directories
        if not any(part.startswith(".") or part == "__pycache__" for part in file_path.parts):
            python_files.append(file_path)

    return python_files


def fix_file_content(file_path: Path) -> Tuple[bool, List[str]]:
    """Fix agent_forge references in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return False, [f"Failed to read {file_path}: {e}"]

    original_content = content
    changes = []

    # Pattern 1: Path references with quotes
    patterns = [
        (r'"core"[\s]*\/[\s]*"agent_forge"', '"core" / "agent_forge"'),
        (r"'core'[\s]*\/[\s]*'agent_forge'", "'core' / 'agent_forge'"),
        (r'"agent_forge"', '"agent_forge"'),
        (r"'agent_forge'", "'agent_forge'"),
        (r"agent_forge", "agent_forge"),
    ]

    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes.append(f"Replaced pattern '{pattern}' -> '{replacement}' ({len(matches)} occurrences)")

    # Write back if changes were made
    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, changes
        except Exception as e:
            return False, [f"Failed to write {file_path}: {e}"]

    return False, []


def main():
    """Main function to fix all agent_forge references."""
    print("Fixing Agent Forge Path References")
    print("=" * 50)

    # Get the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Project root: {project_root}")

    # Find all Python files
    python_files = find_python_files(str(project_root))
    print(f"Found {len(python_files)} Python files to check")

    files_changed = 0
    total_changes = 0
    errors = []

    # Process each file
    for file_path in python_files:
        try:
            changed, changes = fix_file_content(file_path)
            if changed:
                files_changed += 1
                total_changes += len(changes)
                print(f"Fixed: {file_path.relative_to(project_root)}")
                for change in changes[:3]:  # Limit output
                    print(f"   - {change}")
                if len(changes) > 3:
                    print(f"   - ... and {len(changes) - 3} more changes")

        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"   Files processed: {len(python_files)}")
    print(f"   Files changed: {files_changed}")
    print(f"   Total changes: {total_changes}")
    print(f"   Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Limit error output
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   - ... and {len(errors) - 10} more errors")

    print("\nAgent Forge path fixing completed!")

    # Recommend next steps
    print("\nRecommended next steps:")
    print("1. Run tests to verify imports work correctly")
    print("2. Check for any remaining 'agent_forge' references manually")
    print("3. Consider removing the old 'core/agent_forge' directory")


if __name__ == "__main__":
    main()
