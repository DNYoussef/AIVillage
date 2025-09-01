#!/usr/bin/env python3
"""
Quick script to fix import paths in all test files
"""

from pathlib import Path


def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace import paths
    replacements = [
        ("from core.agent_forge.phases.cognate_pretrain", "from agent_forge.phases.cognate_pretrain"),
        ("from core.agent_forge.phases.cognate", "from agent_forge.phases.cognate"),
        ("import core.agent_forge.phases.cognate_pretrain", "import agent_forge.phases.cognate_pretrain"),
    ]

    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
            print(f"  Fixed: {old} -> {new}")

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes needed in {file_path}")


def main():
    """Fix imports in all test files."""
    test_dir = Path(__file__).parent

    test_files = [
        "test_import_validation.py",
        "test_functional_validation.py",
        "test_integration_validation.py",
        "test_file_organization.py",
        "test_error_handling.py",
    ]

    print("Fixing import paths in test files...")

    for test_file in test_files:
        file_path = test_dir / test_file
        if file_path.exists():
            print(f"\nProcessing {test_file}:")
            fix_imports_in_file(file_path)
        else:
            print(f"File not found: {test_file}")

    print("\nImport path fixing complete!")


if __name__ == "__main__":
    main()
