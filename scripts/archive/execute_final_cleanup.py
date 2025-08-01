#!/usr/bin/env python3
"""Execute final cleanup of backup files based on analysis.
This script will safely remove backup files where originals exist.
"""

import json
from datetime import datetime
from pathlib import Path


def load_analysis():
    """Load the analysis and cleanup plan."""
    with open("backup_analysis_report.json") as f:
        return json.load(f)


def execute_cleanup():
    """Execute the cleanup based on conservative rules."""
    analysis = load_analysis()

    cleanup_summary = {
        "cleanup_date": datetime.now().isoformat(),
        "files_removed": [],
        "files_skipped": [],
        "total_space_freed": 0,
    }

    for rec in analysis["recommendations"]:
        file_path = Path(rec["file"])

        # Conservative rule: Only remove if original exists and backup is older than 7 days
        original_path = file_path.with_suffix("")  # Remove .backup extension

        # Check if original exists
        original_exists = original_path.exists()

        # Check file age
        file_age_days = (
            datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        ).days

        # Check if it's a safe removal (original exists and backup is old enough)
        safe_to_remove = original_exists and file_age_days > 7

        if safe_to_remove:
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                cleanup_summary["files_removed"].append(
                    {
                        "file": str(file_path),
                        "size": file_size,
                        "reason": f"Original exists and backup is {file_age_days} days old",
                    }
                )
                cleanup_summary["total_space_freed"] += file_size
            except Exception as e:
                cleanup_summary["files_skipped"].append(
                    {"file": str(file_path), "reason": f"Error removing: {e!s}"}
                )
        else:
            reason = []
            if not original_exists:
                reason.append("Original file missing")
            if file_age_days <= 7:
                reason.append(f"Backup too recent ({file_age_days} days)")

            cleanup_summary["files_skipped"].append(
                {"file": str(file_path), "reason": ", ".join(reason)}
            )

    # Save cleanup summary
    with open("cleanup_summary.json", "w") as f:
        json.dump(cleanup_summary, f, indent=2)

    return cleanup_summary


def validate_cleanup():
    """Validate that cleanup was successful."""
    # Check if any .backup files still exist
    remaining_backups = []
    for file_path in Path().rglob("*.backup*"):
        if "safety_archive" not in str(file_path):
            remaining_backups.append(str(file_path))

    return remaining_backups


def main():
    """Main execution."""
    print("🧹 Executing final cleanup...")

    # Execute cleanup
    summary = execute_cleanup()

    # Print results
    print("\n✅ Cleanup Complete:")
    print(f"  Files removed: {len(summary['files_removed'])}")
    print(f"  Space freed: {summary['total_space_freed']:,} bytes")
    print(f"  Files skipped: {len(summary['files_skipped'])}")

    if summary["files_removed"]:
        print("\n🗑️  Removed files:")
        for file in summary["files_removed"]:
            print(f"  - {file['file']} ({file['size']:,} bytes)")

    if summary["files_skipped"]:
        print("\n⚠️  Skipped files:")
        for file in summary["files_skipped"]:
            print(f"  - {file['file']}: {file['reason']}")

    # Validate
    remaining = validate_cleanup()
    if remaining:
        print(f"\n📋 Remaining backup files: {len(remaining)}")
        for file in remaining:
            print(f"  - {file}")
    else:
        print("\n🎉 All backup files processed successfully!")

    return summary


if __name__ == "__main__":
    main()
