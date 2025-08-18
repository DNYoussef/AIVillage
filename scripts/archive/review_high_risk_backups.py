#!/usr/bin/env python3
"""Review and process HIGH risk source code backups.
This script will analyze each backup file and provide recommendations.
"""

import json
from datetime import datetime
from pathlib import Path


def load_backup_manifest():
    """Load the backup manifest from the safety backup."""
    backup_dirs = list(Path("safety_archive").glob("*"))
    if not backup_dirs:
        msg = "No safety backup found"
        raise ValueError(msg)

    latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
    manifest_path = latest_backup / "backup_manifest.json"

    with open(manifest_path) as f:
        return json.load(f), latest_backup


def analyze_backup_files():
    """Analyze each backup file and provide recommendations."""
    manifest, backup_dir = load_backup_manifest()

    analysis = {
        "analysis_date": datetime.now().isoformat(),
        "backup_directory": str(backup_dir),
        "files_analyzed": len(manifest["backup_log"]),
        "recommendations": [],
    }

    for file_info in manifest["backup_log"]:
        source_path = Path(file_info["source"])
        Path(file_info["backup"])

        # Check if original file exists
        original_exists = source_path.with_suffix("").exists() if source_path.suffix == ".backup" else False

        # Check file age
        file_age_days = (datetime.now() - datetime.fromtimestamp(source_path.stat().st_mtime)).days

        # Check if it's a duplicate backup
        is_duplicate = False
        if original_exists:
            original_size = source_path.with_suffix("").stat().st_size
            backup_size = source_path.stat().st_size
            is_duplicate = original_size == backup_size

        recommendation = {
            "file": str(source_path),
            "size": file_info["size"],
            "age_days": file_age_days,
            "original_exists": original_exists,
            "is_duplicate": is_duplicate,
            "recommendation": determine_recommendation(original_exists, is_duplicate, file_age_days),
        }

        analysis["recommendations"].append(recommendation)

    return analysis


def determine_recommendation(original_exists, is_duplicate, age_days) -> str:
    """Determine what action to take for each backup file."""
    if not original_exists:
        return "KEEP - Original file missing"
    if is_duplicate and age_days > 30:
        return "REMOVE - Duplicate backup older than 30 days"
    if is_duplicate and age_days <= 30:
        return "REVIEW - Recent duplicate backup"
    return "REVIEW - Non-duplicate backup"


def generate_cleanup_plan(analysis):
    """Generate a cleanup plan based on recommendations."""
    plan = {
        "cleanup_date": datetime.now().isoformat(),
        "files_to_remove": [],
        "files_to_keep": [],
        "files_to_review": [],
    }

    for rec in analysis["recommendations"]:
        if rec["recommendation"].startswith("REMOVE"):
            plan["files_to_remove"].append(rec["file"])
        elif rec["recommendation"].startswith("KEEP"):
            plan["files_to_keep"].append(rec["file"])
        else:
            plan["files_to_review"].append({"file": rec["file"], "reason": rec["recommendation"]})

    return plan


def main() -> None:
    """Main execution."""
    print("🔍 Analyzing HIGH risk source code backups...")

    analysis = analyze_backup_files()

    # Save analysis
    with open("backup_analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Generate cleanup plan
    cleanup_plan = generate_cleanup_plan(analysis)

    with open("cleanup_plan.json", "w") as f:
        json.dump(cleanup_plan, f, indent=2)

    # Print summary
    print("\n📊 Analysis Complete:")
    print(f"  Files analyzed: {analysis['files_analyzed']}")
    print(f"  Files to remove: {len(cleanup_plan['files_to_remove'])}")
    print(f"  Files to keep: {len(cleanup_plan['files_to_keep'])}")
    print(f"  Files to review: {len(cleanup_plan['files_to_review'])}")

    if cleanup_plan["files_to_remove"]:
        print("\n🗑️  Files marked for removal:")
        for file in cleanup_plan["files_to_remove"]:
            print(f"  - {file}")

    if cleanup_plan["files_to_review"]:
        print("\n🔍 Files requiring manual review:")
        for item in cleanup_plan["files_to_review"]:
            print(f"  - {item['file']}: {item['reason']}")


if __name__ == "__main__":
    main()
