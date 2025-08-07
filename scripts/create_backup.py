#!/usr/bin/env python3
"""Create a backup of the current codebase structure before restructuring
"""

from datetime import datetime
import json
import os
from pathlib import Path


def create_structure_snapshot():
    """Create a snapshot of the current directory structure."""
    base_path = Path(os.getcwd())

    def scan_directory(path: Path, max_depth=5, current_depth=0):
        """Recursively scan directory structure."""
        if current_depth >= max_depth:
            return {}

        structure = {}

        try:
            for item in path.iterdir():
                if item.name.startswith(".") and item.name not in [".github"]:
                    continue
                if item.name in ["__pycache__", "node_modules", "new_env"]:
                    continue

                if item.is_dir():
                    structure[item.name] = {
                        "type": "directory",
                        "contents": scan_directory(item, max_depth, current_depth + 1),
                    }
                else:
                    structure[item.name] = {
                        "type": "file",
                        "size": item.stat().st_size if item.exists() else 0,
                    }
        except PermissionError:
            structure["_error"] = "Permission denied"

        return structure

    return scan_directory(base_path)


def main():
    print("Creating backup snapshot of current codebase structure...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create structure snapshot
    structure = create_structure_snapshot()

    # Save snapshot
    snapshot_file = f"structure_snapshot_{timestamp}.json"
    with open(snapshot_file, "w") as f:
        json.dump({"timestamp": timestamp, "structure": structure}, f, indent=2)

    print(f"Structure snapshot saved to: {snapshot_file}")

    # Create a list of all important files
    important_files = [
        "pyproject.toml",
        "requirements*.txt",
        "setup.py",
        "Dockerfile*",
        "docker-compose*.yml",
        "*.md",
        "LICENSE",
    ]

    file_list = []
    base_path = Path(os.getcwd())

    for pattern in important_files:
        if "*" in pattern:
            file_list.extend(list(base_path.glob(pattern)))
        else:
            file_path = base_path / pattern
            if file_path.exists():
                file_list.append(file_path)

    backup_info = {
        "timestamp": timestamp,
        "important_files": [str(f.relative_to(base_path)) for f in file_list],
        "directory_count": len(
            [k for k, v in structure.items() if v.get("type") == "directory"]
        ),
        "note": "Backup created before codebase restructuring",
    }

    backup_file = f"backup_info_{timestamp}.json"
    with open(backup_file, "w") as f:
        json.dump(backup_info, f, indent=2)

    print(f"Backup info saved to: {backup_file}")
    print("Backup complete!")

    return backup_info


if __name__ == "__main__":
    main()
