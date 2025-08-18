#!/usr/bin/env python3
"""
AIVillage Architecture Migration Tool

Helps migrate existing code from the flat structure to the new layered architecture.
Provides automated import updating, deprecation detection, and migration guidance.

Usage:
    python migrate_to_new_architecture.py [--scan-dir DIR] [--fix] [--report]
"""

import argparse
import ast
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImportFix:
    """Represents an import that needs to be fixed."""

    file_path: str
    line_number: int
    old_import: str
    new_import: str
    import_type: str  # 'import', 'from_import'


@dataclass
class MigrationResult:
    """Results of migration analysis or fixes."""

    files_scanned: int
    imports_found: int
    imports_fixed: int
    errors: list[str]
    warnings: list[str]
    fixes_applied: list[ImportFix]


class AIVillageCodeMigrator:
    """
    Code migration tool for AIVillage architecture transition.

    Analyzes Python files for old import patterns and provides
    automated fixes for the new layered architecture.
    """

    def __init__(self):
        """Initialize the migration tool."""
        self.import_mappings = self._load_import_mappings()
        self.deprecated_patterns = self._load_deprecated_patterns()

    def _load_import_mappings(self) -> dict[str, str]:
        """Load mappings from old imports to new architecture."""
        return {
            # Core migrations
            "agent_forge": "src.software.agent_forge",
            "production": "src.production",
            "communications": "src.communications",
            "core": "src.core",
            # Specific component migrations
            "mobile": "src.hardware.edge",
            "agents": "src.software.meta_agents",
            "hyperag": "src.hyperag",
            # P2P and transport migrations
            "p2p": "src.core.p2p",
            "transport": "src.transport",
            "federation": "src.federation",
            # Specialized migrations
            "token_economy": "src.token_economy",
            "monitoring": "src.monitoring",
            "security": "src.core.security",
            # Hardware layer migrations
            "android": "src.hardware.android",
            "ios": "src.hardware.ios",
            "edge": "src.hardware.edge",
            # Software layer migrations
            "meta_agents": "src.software.meta_agents",
            "hyper_rag": "src.software.hyper_rag",
            # Architecture layers
            "hardware": "src.hardware",
            "software": "src.software",
            "governance": "src.governance",
        }

    def _load_deprecated_patterns(self) -> list[str]:
        """Load patterns for deprecated imports."""
        return [
            r"from\s+agent_forge\.",
            r"import\s+agent_forge",
            r"from\s+production\.",
            r"import\s+production",
            r"from\s+mobile\.",
            r"import\s+mobile",
            r"from\s+agents\.",
            r"import\s+agents",
        ]

    def scan_directory(self, directory: Path) -> list[Path]:
        """
        Scan directory for Python files to migrate.

        Args:
            directory: Directory to scan

        Returns:
            List of Python files found
        """
        python_files = []

        for path in directory.rglob("*.py"):
            # Skip __pycache__ and .git directories
            if "__pycache__" in str(path) or ".git" in str(path):
                continue
            # Skip files in src/ directory (already migrated)
            if "src/" in str(path):
                continue

            python_files.append(path)

        logger.info(f"Found {len(python_files)} Python files to scan")
        return python_files

    def analyze_file(self, file_path: Path) -> list[ImportFix]:
        """
        Analyze a Python file for migration opportunities.

        Args:
            file_path: Path to Python file to analyze

        Returns:
            List of ImportFix objects for needed changes
        """
        fixes = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the AST to find imports
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle "import module" statements
                    for alias in node.names:
                        old_name = alias.name
                        if old_name in self.import_mappings:
                            new_name = self.import_mappings[old_name]
                            fixes.append(
                                ImportFix(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    old_import=f"import {old_name}",
                                    new_import=f"import {new_name}",
                                    import_type="import",
                                )
                            )

                elif isinstance(node, ast.ImportFrom):
                    # Handle "from module import ..." statements
                    if node.module and node.module in self.import_mappings:
                        old_module = node.module
                        new_module = self.import_mappings[old_module]

                        # Get the import items
                        items = [alias.name for alias in node.names]
                        items_str = ", ".join(items)

                        fixes.append(
                            ImportFix(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                old_import=f"from {old_module} import {items_str}",
                                new_import=f"from {new_module} import {items_str}",
                                import_type="from_import",
                            )
                        )

            # Also check for relative imports that might need updates
            fixes.extend(self._check_relative_imports(file_path, content))

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

        return fixes

    def _check_relative_imports(self, file_path: Path, content: str) -> list[ImportFix]:
        """Check for relative imports that might need updating."""
        fixes = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # Check for patterns that suggest old architecture usage
            for pattern in self.deprecated_patterns:
                if re.search(pattern, line):
                    # Try to suggest a fix based on the pattern
                    suggested_fix = self._suggest_import_fix(line)
                    if suggested_fix:
                        fixes.append(
                            ImportFix(
                                file_path=str(file_path),
                                line_number=i,
                                old_import=line,
                                new_import=suggested_fix,
                                import_type="pattern_match",
                            )
                        )

        return fixes

    def _suggest_import_fix(self, import_line: str) -> str | None:
        """Suggest a fix for an import line."""
        for old_pattern, new_module in self.import_mappings.items():
            if old_pattern in import_line:
                return import_line.replace(old_pattern, new_module)
        return None

    def apply_fixes(self, fixes: list[ImportFix]) -> MigrationResult:
        """
        Apply import fixes to files.

        Args:
            fixes: List of ImportFix objects to apply

        Returns:
            MigrationResult with results of applying fixes
        """
        result = MigrationResult(
            files_scanned=len(set(fix.file_path for fix in fixes)),
            imports_found=len(fixes),
            imports_fixed=0,
            errors=[],
            warnings=[],
            fixes_applied=[],
        )

        # Group fixes by file
        fixes_by_file = {}
        for fix in fixes:
            if fix.file_path not in fixes_by_file:
                fixes_by_file[fix.file_path] = []
            fixes_by_file[fix.file_path].append(fix)

        # Apply fixes file by file
        for file_path, file_fixes in fixes_by_file.items():
            try:
                self._apply_fixes_to_file(file_path, file_fixes, result)
            except Exception as e:
                error_msg = f"Error applying fixes to {file_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        return result

    def _apply_fixes_to_file(self, file_path: str, fixes: list[ImportFix], result: MigrationResult):
        """Apply fixes to a single file."""
        # Read the file
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Sort fixes by line number (reverse order to avoid line number shifts)
        fixes.sort(key=lambda x: x.line_number, reverse=True)

        # Apply each fix
        for fix in fixes:
            try:
                line_idx = fix.line_number - 1  # Convert to 0-based index
                if 0 <= line_idx < len(lines):
                    old_line = lines[line_idx].rstrip("\n")

                    # Replace the old import with the new one
                    if fix.old_import.strip() in old_line:
                        new_line = old_line.replace(fix.old_import.strip(), fix.new_import)
                        lines[line_idx] = new_line + "\n"

                        result.imports_fixed += 1
                        result.fixes_applied.append(fix)

                        logger.debug(f"Fixed: {fix.old_import} -> {fix.new_import}")
                    else:
                        warning_msg = f"Could not find exact match for import in {file_path}:{fix.line_number}"
                        logger.warning(warning_msg)
                        result.warnings.append(warning_msg)

            except Exception as e:
                error_msg = f"Error applying fix {fix.old_import} -> {fix.new_import}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Write the updated file back
        if result.fixes_applied:  # Only write if we made changes
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"Updated {file_path} with {len([f for f in fixes if f.file_path == file_path])} fixes")

    def generate_migration_report(
        self,
        scan_results: list[ImportFix],
        migration_result: MigrationResult | None = None,
    ) -> str:
        """
        Generate a comprehensive migration report.

        Args:
            scan_results: Results from scanning files
            migration_result: Optional results from applying fixes

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("AIVillage Architecture Migration Report")
        report.append("=" * 70)

        # Summary
        files_with_issues = len(set(fix.file_path for fix in scan_results))
        total_imports = len(scan_results)

        report.append(f"Files scanned: {files_with_issues}")
        report.append(f"Import issues found: {total_imports}")

        if migration_result:
            report.append(f"Import fixes applied: {migration_result.imports_fixed}")
            report.append(f"Errors: {len(migration_result.errors)}")
            report.append(f"Warnings: {len(migration_result.warnings)}")

        report.append("")

        # Group issues by type
        import_issues = {}
        for fix in scan_results:
            old_module = self._extract_module_name(fix.old_import)
            if old_module not in import_issues:
                import_issues[old_module] = []
            import_issues[old_module].append(fix)

        # Report issues by module
        report.append("Import Issues by Module:")
        report.append("-" * 40)

        for module, fixes in sorted(import_issues.items()):
            report.append(f"\n{module}: ({len(fixes)} occurrences)")

            # Show a few examples
            for fix in fixes[:3]:
                file_name = os.path.basename(fix.file_path)
                report.append(f"  {file_name}:{fix.line_number} - {fix.old_import}")

            if len(fixes) > 3:
                report.append(f"  ... and {len(fixes) - 3} more")

        # Migration guidance
        report.append("\n" + "=" * 70)
        report.append("Migration Guidance:")
        report.append("=" * 70)

        report.append("\n1. Update Import Statements:")
        for old_module, new_module in self.import_mappings.items():
            if any(old_module in fix.old_import for fix in scan_results):
                report.append(f"   {old_module} -> {new_module}")

        report.append("\n2. Architecture Changes:")
        report.append("   - Mobile components moved to src.hardware.edge")
        report.append("   - Agents moved to src.software.meta_agents")
        report.append("   - Core utilities moved to src.core")
        report.append("   - Production code organized under src.production")

        report.append("\n3. Next Steps:")
        if migration_result and migration_result.imports_fixed > 0:
            report.append("   ✓ Import fixes have been applied")
            report.append("   - Test your code to ensure it works with new imports")
            report.append("   - Check for any remaining compatibility issues")
        else:
            report.append("   - Run with --fix to automatically update imports")
            report.append("   - Test thoroughly after migration")

        report.append("\n4. Compatibility Bridges:")
        report.append("   - Legacy import bridges are available during transition")
        report.append("   - Add 'import src.core.compatibility' to enable bridges")

        if migration_result and migration_result.errors:
            report.append("\nErrors Encountered:")
            report.append("-" * 30)
            for error in migration_result.errors:
                report.append(f"  - {error}")

        if migration_result and migration_result.warnings:
            report.append("\nWarnings:")
            report.append("-" * 30)
            for warning in migration_result.warnings:
                report.append(f"  - {warning}")

        return "\n".join(report)

    def _extract_module_name(self, import_statement: str) -> str:
        """Extract the main module name from an import statement."""
        # Simple extraction - could be made more robust
        if "from " in import_statement:
            # "from module import ..."
            match = re.search(r"from\s+(\S+)", import_statement)
            if match:
                return match.group(1).split(".")[0]
        elif "import " in import_statement:
            # "import module"
            match = re.search(r"import\s+(\S+)", import_statement)
            if match:
                return match.group(1).split(".")[0]

        return "unknown"


def main():
    """Main migration tool entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate AIVillage code to new layered architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_to_new_architecture.py --scan-dir /path/to/project --report
  python migrate_to_new_architecture.py --scan-dir /path/to/project --fix
  python migrate_to_new_architecture.py --report  # Scans current directory
        """,
    )

    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to scan for Python files (default: current directory)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes to files (modifies files in-place)",
    )

    parser.add_argument("--report", action="store_true", help="Generate and display migration report")

    parser.add_argument("--output", type=Path, help="Output file for migration report")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create migrator
    migrator = AIVillageCodeMigrator()

    try:
        # Scan directory for Python files
        logger.info(f"Scanning directory: {args.scan_dir}")
        python_files = migrator.scan_directory(args.scan_dir)

        if not python_files:
            print("No Python files found to migrate.")
            return 0

        # Analyze all files for migration opportunities
        all_fixes = []
        for file_path in python_files:
            fixes = migrator.analyze_file(file_path)
            all_fixes.extend(fixes)

        if not all_fixes:
            print("No import issues found. Your code appears to be up-to-date!")
            return 0

        print(f"Found {len(all_fixes)} import issues across {len(set(fix.file_path for fix in all_fixes))} files")

        # Apply fixes if requested
        migration_result = None
        if args.fix:
            print("Applying fixes...")
            migration_result = migrator.apply_fixes(all_fixes)
            print(f"Applied {migration_result.imports_fixed} fixes")

            if migration_result.errors:
                print(f"Encountered {len(migration_result.errors)} errors")
            if migration_result.warnings:
                print(f"Generated {len(migration_result.warnings)} warnings")

        # Generate report if requested
        if args.report or args.output:
            report = migrator.generate_migration_report(all_fixes, migration_result)

            if args.output:
                args.output.write_text(report)
                print(f"Migration report saved to: {args.output}")
            else:
                print(report)

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
