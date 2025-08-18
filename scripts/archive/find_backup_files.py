#!/usr/bin/env python3
"""Find all .backup files in the workspace and categorize them by risk level."""

import json
from datetime import datetime
from pathlib import Path


def find_backup_files():
    """Find all .backup files and categorize them."""
    workspace = Path()
    backup_files = []

    # Find all .backup files
    for file_path in workspace.rglob("*.backup*"):
        if "safety_archive" not in str(file_path):
            backup_files.append(
                {
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "risk_level": categorize_risk(file_path),
                }
            )

    return backup_files


def categorize_risk(file_path) -> str:
    """Categorize backup files by risk level."""
    path_str = str(file_path).lower()

    # HIGH risk: Source code backups
    if any(ext in path_str for ext in [".py.backup", ".js.backup", ".ts.backup", ".java.backup"]):
        return "HIGH"

    # MEDIUM risk: Configuration backups
    if any(name in path_str for name in ["config", "yaml", "json", "ini", "toml"]):
        return "MEDIUM"

    # LOW risk: Test and migration backups
    if any(name in path_str for name in ["test", "migration", "migrate", "backup_test"]):
        return "LOW"

    # Default to MEDIUM for unknown types
    return "MEDIUM"


def main() -> None:
    """Main execution."""
    print("Scanning for backup files...")
    backup_files = find_backup_files()

    # Group by risk level
    risk_groups = {
        "LOW": [f for f in backup_files if f["risk_level"] == "LOW"],
        "MEDIUM": [f for f in backup_files if f["risk_level"] == "MEDIUM"],
        "HIGH": [f for f in backup_files if f["risk_level"] == "HIGH"],
    }

    # Save inventory
    inventory = {
        "scan_date": datetime.now().isoformat(),
        "total_files": len(backup_files),
        "risk_breakdown": {
            "LOW": len(risk_groups["LOW"]),
            "MEDIUM": len(risk_groups["MEDIUM"]),
            "HIGH": len(risk_groups["HIGH"]),
        },
        "files": backup_files,
        "risk_groups": risk_groups,
    }

    with open("backup_files_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    # Print summary
    print(f"\nFound {len(backup_files)} backup files:")
    print(f"  LOW risk: {len(risk_groups['LOW'])} files")
    print(f"  MEDIUM risk: {len(risk_groups['MEDIUM'])} files")
    print(f"  HIGH risk: {len(risk_groups['HIGH'])} files")

    # Print files by category
    for risk, files in risk_groups.items():
        if files:
            print(f"\n{risk} RISK FILES:")
            for file in files:
                print(f"  - {file['path']} ({file['size']} bytes)")


if __name__ == "__main__":
    main()
