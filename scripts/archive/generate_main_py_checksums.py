#!/usr/bin/env python3
"""Generate checksums for main.py files to identify duplicates."""

import csv
import hashlib
import os
from pathlib import Path


def calculate_sha256(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def get_file_size(file_path):
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0


def main() -> None:
    # Read filtered locations
    try:
        with open("main_py_locations_filtered.txt") as f:
            locations = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("main_py_locations_filtered.txt not found")
        return

    # Generate checksums and details
    results = []
    for location in locations:
        full_path = Path(location)
        if full_path.exists():
            checksum = calculate_sha256(full_path)
            size = get_file_size(full_path)
            results.append(
                {
                    "path": str(location),
                    "full_path": str(full_path),
                    "checksum": checksum,
                    "size_bytes": size,
                }
            )

    # Write checksums to CSV
    with open("main_py_checksums.csv", "w", newline="") as csvfile:
        fieldnames = ["path", "full_path", "checksum", "size_bytes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Write detailed file listing
    with open("main_py_details.txt", "w") as f:
        f.write("main.py files analysis:\n")
        f.write("=" * 50 + "\n\n")

        for result in results:
            f.write(f"Path: {result['path']}\n")
            f.write(f"Full Path: {result['full_path']}\n")
            f.write(f"Size: {result['size_bytes']} bytes\n")
            f.write(f"SHA256: {result['checksum']}\n")
            f.write("-" * 30 + "\n")

    # Check for duplicates
    checksum_groups = {}
    for result in results:
        checksum = result["checksum"]
        if checksum not in checksum_groups:
            checksum_groups[checksum] = []
        checksum_groups[checksum].append(result)

    # Write duplicate analysis
    with open("main_py_duplicates.txt", "w") as f:
        f.write("Duplicate main.py files analysis:\n")
        f.write("=" * 50 + "\n\n")

        duplicates = {k: v for k, v in checksum_groups.items() if len(v) > 1}

        if duplicates:
            f.write(f"Found {len(duplicates)} groups of identical files:\n\n")
            for checksum, files in duplicates.items():
                f.write(f"Checksum: {checksum}\n")
                f.write(f"Files ({len(files)} total):\n")
                for file_info in files:
                    f.write(f"  - {file_info['path']} ({file_info['size_bytes']} bytes)\n")
                f.write("\n")
        else:
            f.write("No identical main.py files found.\n")

    print(f"Processed {len(results)} main.py files")
    print("Results saved to:")
    print("- main_py_checksums.csv")
    print("- main_py_details.txt")
    print("- main_py_duplicates.txt")


if __name__ == "__main__":
    main()
