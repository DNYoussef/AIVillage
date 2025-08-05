#!/usr/bin/env python3
"""Script to update all config/ paths to configs/ paths"""

from pathlib import Path
import re


def update_file(file_path: Path, patterns_to_replace: list[tuple[str, str]]):
    """Update file with pattern replacements"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        for old_pattern, new_pattern in patterns_to_replace:
            content = re.sub(old_pattern, new_pattern, content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main function to update all config paths"""
    root_dir = Path()

    # Patterns to replace
    patterns = [
        (r'\bconfig/([^/\s"\']+\.ya?ml)', r"configs/\1"),
        (r'\bconfig/([^/\s"\']+\.json)', r"configs/\1"),
        (r"config/gdc_rules\.yaml", r"configs/gdc_rules.yaml"),
        (r"config/hyperag_mcp\.yaml", r"configs/hyperag_mcp.yaml"),
        (r"config/compression\.yaml", r"configs/compression.yaml"),
        (r"config/retrieval\.yaml", r"configs/retrieval.yaml"),
        (r"config/scanner_config\.json", r"configs/scanner_config.yaml"),
    ]

    # File types to update
    file_extensions = [".py", ".md", ".yml", ".yaml", ".json", ".sh", ".env.mcp"]

    updated_files = []

    # Walk through all files
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and any(str(file_path).endswith(ext) for ext in file_extensions):
            # Skip files in certain directories
            if any(part in str(file_path) for part in [".git", "__pycache__", "node_modules", ".pytest_cache"]):
                continue

            if update_file(file_path, patterns):
                updated_files.append(str(file_path))

    print(f"\nUpdated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
