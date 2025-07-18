#!/usr/bin/env python3
"""Create a safety backup of all .backup files before cleanup."""

from datetime import datetime
import json
from pathlib import Path
import shutil


def create_safety_backup():
    """Create timestamped backup of all .backup files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("safety_archive") / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Load inventory
    with open("backup_files_inventory.json") as f:
        inventory = json.load(f)

    backup_count = 0
    backup_log = []

    for file_info in inventory["files"]:
        source_path = Path(file_info["path"])
        if source_path.exists():
            # Create relative path structure in backup
            relative_path = source_path.relative_to(Path())
            dest_path = backup_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(source_path, dest_path)
            backup_count += 1

            backup_log.append(
                {
                    "source": str(source_path),
                    "backup": str(dest_path),
                    "size": file_info["size"],
                }
            )

    # Save backup manifest
    manifest = {
        "backup_date": datetime.now().isoformat(),
        "backup_directory": str(backup_dir),
        "files_backed_up": backup_count,
        "backup_log": backup_log,
    }

    with open(backup_dir / "backup_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Safety backup created: {backup_dir}")
    print(f"📁 Backed up {backup_count} files")

    return str(backup_dir)


if __name__ == "__main__":
    backup_dir = create_safety_backup()
    print("\nNext step: Proceed with Phase 2 - Remove LOW risk files")
