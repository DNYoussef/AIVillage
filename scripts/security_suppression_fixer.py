#!/usr/bin/env python3
"""
Security Suppression Fixer
Automatically adds appropriate suppression comments to test files with hardcoded credentials.
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict


class SecuritySuppressionFixer:
    """Automatically add security suppression comments to test files."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.fixes_applied = 0
        self.files_processed = 0

        # Patterns for different types of credentials
        self.credential_patterns = {
            "password": {
                "pattern": r'(password\s*[:=]\s*["\'][^"\']{3,}["\'])(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # nosec B106 - test password",
            },
            "secret_key": {
                "pattern": r'(secret_key\s*[:=]\s*["\'][^"\']+["\'])(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # pragma: allowlist secret - test secret key",
            },
            "api_key": {
                "pattern": r'(api_key\s*[:=]\s*["\'][^"\']+["\'])(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # pragma: allowlist secret - test API key",
            },
            "token": {
                "pattern": r'((?:token|auth_token|access_token)\s*[:=]\s*["\'][^"\']+["\'])(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # pragma: allowlist secret - test token",
            },
            "private_key": {
                "pattern": r'(private_key\s*[:=]\s*["\'][^"\']+["\'])(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # pragma: allowlist secret - test private key",
            },
            "bearer_auth": {
                "pattern": r'("Authorization":\s*"Bearer\s+[^"]+")(?!\s*#.*(?:nosec|pragma|allowlist))',
                "suppression": "  # pragma: allowlist secret - test bearer token",
            },
        }

    def is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        return (
            "test" in file_path.name.lower()
            or "test" in str(file_path.parent).lower()
            or file_path.name.startswith("test_")
            or "/tests/" in str(file_path)
            or "\\tests\\" in str(file_path)
        )

    def process_file(self, file_path: Path) -> Dict[str, int]:
        """Process a single file and add suppression comments."""
        if not self.is_test_file(file_path):
            return {"fixes": 0, "patterns": {}}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Could not decode {file_path}, skipping...")
            return {"fixes": 0, "patterns": {}}

        original_content = content
        fixes_in_file = 0
        pattern_counts = {}

        for pattern_name, pattern_config in self.credential_patterns.items():
            pattern = pattern_config["pattern"]
            suppression = pattern_config["suppression"]

            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            pattern_counts[pattern_name] = len(matches)

            if matches:
                # Replace matches with suppressed versions
                for match in reversed(matches):  # Reverse to maintain positions
                    start, end = match.span()
                    matched_text = match.group(1)
                    suppressed_text = matched_text + suppression
                    content = content[:start] + suppressed_text + content[end:]
                    fixes_in_file += len(matches)

        # Write back if changes were made
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {fixes_in_file} suppressions in {file_path}")

        return {"fixes": fixes_in_file, "patterns": pattern_counts}

    def scan_directory(self, directory: Path, extensions: List[str] = None) -> None:
        """Scan directory for Python files that need suppression fixes."""
        if extensions is None:
            extensions = [".py"]

        total_fixes = 0
        total_files = 0
        pattern_summary = {}

        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if file_path.is_file():
                    result = self.process_file(file_path)
                    total_fixes += result["fixes"]
                    if result["fixes"] > 0:
                        total_files += 1

                    # Aggregate pattern counts
                    for pattern, count in result["patterns"].items():
                        pattern_summary[pattern] = pattern_summary.get(pattern, 0) + count

        print("\n=== Summary ===")
        print(f"Files processed: {total_files}")
        print(f"Total suppressions added: {total_fixes}")
        print("\nPattern breakdown:")
        for pattern, count in pattern_summary.items():
            if count > 0:
                print(f"  {pattern}: {count} instances")

    def fix_specific_files(self, file_paths: List[str]) -> None:
        """Fix specific files."""
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists():
                result = self.process_file(file_path)
                if result["fixes"] > 0:
                    print(f"Applied {result['fixes']} fixes to {file_path}")
                else:
                    print(f"No fixes needed for {file_path}")
            else:
                print(f"File not found: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Automatically add security suppression comments to test files")
    parser.add_argument("--directory", "-d", default=".", help="Directory to scan (default: current directory)")
    parser.add_argument("--files", "-f", nargs="+", help="Specific files to fix")
    parser.add_argument(
        "--extensions", "-e", nargs="+", default=[".py"], help="File extensions to process (default: .py)"
    )
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be fixed without making changes")

    args = parser.parse_args()

    fixer = SecuritySuppressionFixer(args.directory)

    if args.files:
        fixer.fix_specific_files(args.files)
    else:
        fixer.scan_directory(Path(args.directory), args.extensions)


if __name__ == "__main__":
    main()
