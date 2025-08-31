#!/usr/bin/env python3
"""
Clean Architecture Migration Tool - Refactored

Migrates existing codebase to clean architecture patterns using
modular components for maintainability and single responsibility.

Usage:
    python scripts/architecture/migrate_to_clean_architecture.py [--dry-run] [--target-layer]
"""

import argparse
import ast
from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MigrationPlan:
    """Represents a migration plan for a file or component."""

    source_path: Path
    target_path: Path
    migration_type: str  # "move", "refactor", "split", "extract"
    reason: str
    dependencies: list[str]
    estimated_effort: str  # "low", "medium", "high"


@dataclass
class RefactoringAction:
    """Represents a code refactoring action."""

    file_path: Path
    action_type: str  # "extract_class", "extract_method", "introduce_interface"
    target_component: str
    description: str


class MigrationPlanner:
    """Plans migration steps based on current architecture analysis."""

    def __init__(self, project_root: Path, config: dict[str, Any]):
        self.project_root = project_root
        self.config = config
        self.migration_plans = []

    def analyze_current_structure(self) -> dict[str, Any]:
        """Analyze current project structure and identify migration needs."""
        analysis = {"misplaced_files": [], "large_files": [], "coupling_issues": [], "layer_violations": []}

        # Find misplaced files based on naming patterns
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            relative_path = py_file.relative_to(self.project_root)
            file_name = py_file.name.lower()

            # Check for repository patterns outside infrastructure
            if "repository" in file_name and not str(relative_path).startswith("infrastructure"):
                analysis["misplaced_files"].append(
                    {
                        "file": str(relative_path),
                        "suggested_location": "infrastructure",
                        "reason": "Repository pattern should be in infrastructure layer",
                    }
                )

            # Check for service patterns outside core
            if "service" in file_name and not str(relative_path).startswith(("core", "apps")):
                analysis["misplaced_files"].append(
                    {
                        "file": str(relative_path),
                        "suggested_location": "core",
                        "reason": "Service pattern should be in core or apps layer",
                    }
                )

            # Check file size
            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    line_count = sum(1 for _ in f)

                if line_count > 500:
                    analysis["large_files"].append(
                        {
                            "file": str(relative_path),
                            "lines": line_count,
                            "reason": "File exceeds recommended size limit",
                        }
                    )
            except Exception as e:
                logging.debug(f"Failed to analyze file {py_file}: {e}")

        return analysis

    def create_migration_plans(self) -> list[MigrationPlan]:
        """Create detailed migration plans based on analysis."""
        analysis = self.analyze_current_structure()
        plans = []

        # Plan file moves
        for misplaced in analysis["misplaced_files"]:
            source_path = self.project_root / misplaced["file"]
            target_dir = self.project_root / misplaced["suggested_location"]
            target_path = target_dir / source_path.name

            plans.append(
                MigrationPlan(
                    source_path=source_path,
                    target_path=target_path,
                    migration_type="move",
                    reason=misplaced["reason"],
                    dependencies=self._analyze_dependencies(source_path),
                    estimated_effort="low",
                )
            )

        # Plan file splits for large files
        for large_file in analysis["large_files"]:
            source_path = self.project_root / large_file["file"]

            plans.append(
                MigrationPlan(
                    source_path=source_path,
                    target_path=source_path,  # Will be split into multiple files
                    migration_type="split",
                    reason=f"File has {large_file['lines']} lines, needs splitting",
                    dependencies=self._analyze_dependencies(source_path),
                    estimated_effort="high",
                )
            )

        self.migration_plans = plans
        return plans

    def _analyze_dependencies(self, file_path: Path) -> list[str]:
        """Analyze dependencies for a given file."""
        dependencies = []
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_internal_import(alias.name):
                            dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and self._is_internal_import(node.module):
                        dependencies.append(node.module)
        except Exception as e:
            logging.debug(f"Failed to extract dependencies from {file_path}: {e}")

        return dependencies

    def _is_internal_import(self, module_name: str) -> bool:
        """Check if import is internal to the project."""
        return module_name.startswith(("core", "infrastructure", "apps", "integrations"))


class CodeTransformer:
    """Handles code transformation and refactoring operations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.refactoring_actions = []

    def extract_interfaces(self, file_path: Path) -> list[RefactoringAction]:
        """Extract interfaces from concrete implementations."""
        actions = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tree = ast.parse(content)

            # Find classes that could have interfaces extracted
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self._should_extract_interface(node):
                        actions.append(
                            RefactoringAction(
                                file_path=file_path,
                                action_type="extract_interface",
                                target_component=f"I{node.name}",
                                description=f"Extract interface for {node.name} class",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not analyze {file_path} for interface extraction: {e}")

        return actions

    def _should_extract_interface(self, class_node: ast.ClassDef) -> bool:
        """Determine if a class should have an interface extracted."""
        # Simple heuristic: classes with many public methods
        public_methods = [
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]

        return len(public_methods) >= 3

    def split_large_file(self, file_path: Path, dry_run: bool = True) -> list[Path]:
        """Split a large file into smaller, focused modules."""
        if dry_run:
            logger.info(f"DRY RUN: Would split {file_path}")
            return []

        new_files = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tree = ast.parse(content)

            # Group related classes and functions
            components = self._group_components(tree)

            # Create new files for each component group
            for i, (group_name, group_nodes) in enumerate(components.items()):
                new_file_path = file_path.with_name(f"{file_path.stem}_{group_name}.py")

                # Generate content for new file
                new_content = self._generate_file_content(group_nodes, content)

                with open(new_file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                new_files.append(new_file_path)
                logger.info(f"Created {new_file_path}")

        except Exception as e:
            logger.error(f"Failed to split {file_path}: {e}")

        return new_files

    def _group_components(self, tree: ast.AST) -> dict[str, list[ast.AST]]:
        """Group related components for splitting."""
        groups = {"models": [], "services": [], "utilities": [], "interfaces": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Classify based on naming patterns
                class_name = node.name.lower()
                if "model" in class_name or "entity" in class_name:
                    groups["models"].append(node)
                elif "service" in class_name or "manager" in class_name:
                    groups["services"].append(node)
                elif "interface" in class_name or node.name.startswith("I"):
                    groups["interfaces"].append(node)
                else:
                    groups["utilities"].append(node)
            elif isinstance(node, ast.FunctionDef) and not hasattr(node, "decorator_list"):
                groups["utilities"].append(node)

        # Remove empty groups
        return {name: nodes for name, nodes in groups.items() if nodes}

    def _generate_file_content(self, nodes: list[ast.AST], original_content: str) -> str:
        """Generate content for a new file from AST nodes."""
        # This is a simplified implementation
        # In a real scenario, you'd want more sophisticated code generation
        imports = "# Auto-generated imports\nimport logging\nfrom typing import Any, Dict, List\n\n"

        # For now, just return a basic structure
        class_definitions = ""
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                class_definitions += f"class {node.name}:\n    pass\n\n"
            elif isinstance(node, ast.FunctionDef):
                class_definitions += f"def {node.name}():\n    pass\n\n"

        return imports + class_definitions


class FileReorganizer:
    """Handles file system operations for migration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def execute_migration_plan(self, plan: MigrationPlan, dry_run: bool = True) -> bool:
        """Execute a single migration plan."""
        if dry_run:
            logger.info(f"DRY RUN: {plan.migration_type.upper()} {plan.source_path} -> {plan.target_path}")
            return True

        try:
            if plan.migration_type == "move":
                return self._move_file(plan.source_path, plan.target_path)
            elif plan.migration_type == "split":
                return self._prepare_for_split(plan.source_path)
            else:
                logger.warning(f"Unknown migration type: {plan.migration_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to execute migration plan: {e}")
            return False

    def _move_file(self, source: Path, target: Path) -> bool:
        """Move a file to a new location."""
        # Create target directory if it doesn't exist
        target.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(source), str(target))
        logger.info(f"Moved {source} to {target}")

        return True

    def _prepare_for_split(self, file_path: Path) -> bool:
        """Prepare a file for splitting (backup and mark)."""
        backup_path = file_path.with_suffix(".py.backup")
        shutil.copy2(str(file_path), str(backup_path))
        logger.info(f"Created backup: {backup_path}")
        return True

    def create_layer_structure(self, dry_run: bool = True) -> bool:
        """Create clean architecture layer structure."""
        layer_dirs = [
            "core/domain",
            "core/application/services",
            "core/application/interfaces",
            "infrastructure/repositories",
            "infrastructure/external",
            "apps/web",
            "apps/cli",
            "integrations/external",
        ]

        if dry_run:
            logger.info("DRY RUN: Would create layer structure:")
            for layer_dir in layer_dirs:
                logger.info(f"  - {layer_dir}/")
            return True

        for layer_dir in layer_dirs:
            full_path = self.project_root / layer_dir
            full_path.mkdir(parents=True, exist_ok=True)

            # Create __init__.py files
            init_file = full_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Clean architecture layer module."""\n')

            logger.info(f"Created layer directory: {layer_dir}")

        return True


class CleanArchitectureMigrator:
    """Main migration coordinator - simplified facade."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

        # Load configuration with defaults
        config_path = project_root / "config" / "architecture" / "migration_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "migration": {"backup_enabled": True, "validate_after_migration": True, "create_layer_structure": True}
            }

        # Initialize component migrators
        self.planner = MigrationPlanner(project_root, self.config)
        self.transformer = CodeTransformer(project_root)
        self.reorganizer = FileReorganizer(project_root)

    def run_migration(self, dry_run: bool = True, target_layer: str = None) -> bool:
        """Run complete migration to clean architecture."""
        logger.info("Starting clean architecture migration...")

        if dry_run:
            logger.info("Running in DRY RUN mode - no changes will be made")

        # Step 1: Create layer structure
        if self.config.get("migration", {}).get("create_layer_structure", True):
            logger.info("Creating clean architecture layer structure...")
            self.reorganizer.create_layer_structure(dry_run)

        # Step 2: Analyze current structure and create migration plans
        logger.info("Analyzing current structure...")
        migration_plans = self.planner.create_migration_plans()

        if not migration_plans:
            logger.info("No migration needed - architecture is already clean!")
            return True

        logger.info(f"Created {len(migration_plans)} migration plans")

        # Step 3: Execute migration plans
        successful_migrations = 0

        for plan in migration_plans:
            if target_layer and target_layer not in str(plan.target_path):
                continue

            logger.info(f"Executing: {plan.reason}")

            if self.reorganizer.execute_migration_plan(plan, dry_run):
                successful_migrations += 1
            else:
                logger.warning(f"Failed to execute migration plan for {plan.source_path}")

        # Step 4: Update imports and references (dry run only logs)
        if not dry_run:
            self._update_import_references(migration_plans)
        else:
            logger.info("DRY RUN: Would update import references after migration")

        logger.info(f"Migration complete: {successful_migrations}/{len(migration_plans)} successful")

        return successful_migrations == len(migration_plans)

    def generate_migration_report(self, output_file: Path = None) -> dict[str, Any]:
        """Generate migration analysis and planning report."""
        analysis = self.planner.analyze_current_structure()
        migration_plans = self.planner.create_migration_plans()

        report = {
            "timestamp": str(Path.cwd()),  # Placeholder
            "analysis": analysis,
            "migration_plans": [
                {
                    "source": str(plan.source_path.relative_to(self.project_root)),
                    "target": str(plan.target_path.relative_to(self.project_root)),
                    "type": plan.migration_type,
                    "reason": plan.reason,
                    "effort": plan.estimated_effort,
                    "dependencies": len(plan.dependencies),
                }
                for plan in migration_plans
            ],
            "summary": {
                "total_files_to_migrate": len(migration_plans),
                "misplaced_files": len(analysis["misplaced_files"]),
                "large_files": len(analysis["large_files"]),
                "estimated_effort": self._calculate_total_effort(migration_plans),
            },
        }

        if output_file:
            import json

            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Migration report saved to {output_file}")

        return report

    def _update_import_references(self, migration_plans: list[MigrationPlan]):
        """Update import statements after files have been moved."""
        # This would involve updating all import statements across the codebase
        # For now, just log what would be updated
        for plan in migration_plans:
            if plan.migration_type == "move":
                logger.info(f"Would update imports for moved file: {plan.source_path}")

    def _calculate_total_effort(self, plans: list[MigrationPlan]) -> str:
        """Calculate total estimated effort for migration."""
        effort_scores = {"low": 1, "medium": 3, "high": 5}
        total_score = sum(effort_scores.get(plan.estimated_effort, 1) for plan in plans)

        if total_score <= 5:
            return "low"
        elif total_score <= 15:
            return "medium"
        else:
            return "high"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean Architecture Migration Tool")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in dry-run mode (default)")
    parser.add_argument("--execute", action="store_true", help="Execute migration (overrides --dry-run)")
    parser.add_argument("--target-layer", type=str, help="Target specific layer for migration")
    parser.add_argument("--report", type=Path, help="Generate migration report")

    args = parser.parse_args()

    # Determine if this is a dry run
    dry_run = not args.execute

    # Run migration
    migrator = CleanArchitectureMigrator(args.project_root)

    # Generate report if requested
    if args.report:
        migrator.generate_migration_report(args.report)

    # Run migration
    success = migrator.run_migration(dry_run, args.target_layer)

    if success:
        logger.info("Migration completed successfully!")
        return 0
    else:
        logger.error("Migration completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
