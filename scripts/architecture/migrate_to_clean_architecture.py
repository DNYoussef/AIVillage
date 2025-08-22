#!/usr/bin/env python3
"""
Clean Architecture Migration Script

This script migrates the AIVillage codebase from the current packages/ structure
to a clean architecture with proper layer separation.

Usage:
    python scripts/architecture/migrate_to_clean_architecture.py --phase <1-5> [--dry-run]
"""

import argparse
import ast
import logging
import shutil
import sys
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CleanArchitectureMigrator:
    """Handles migration to clean architecture"""

    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.packages_dir = project_root / "packages"

        # Load configuration
        config_path = project_root / "config" / "architecture" / "clean_architecture_rules.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Migration mappings
        self.migration_map = self._build_migration_map()

        # Track moved files for rollback
        self.moved_files = []

    def _build_migration_map(self) -> dict[str, str]:
        """Build mapping of source to destination directories"""

        map_config = self.config["architecture"]["migration_rules"]

        migration_map = {}

        # Apps layer mappings
        for source in map_config.get("move_to_apps", []):
            if source.endswith("/"):
                source = source[:-1]
            migration_map[source] = f"apps/{self._determine_app_type(source)}"

        # Core layer mappings
        for source in map_config.get("move_to_core", []):
            if source.endswith("/"):
                source = source[:-1]
            migration_map[source] = f"core/{self._determine_core_type(source)}"

        # Infrastructure layer mappings
        for source in map_config.get("move_to_infrastructure", []):
            if source.endswith("/"):
                source = source[:-1]
            migration_map[source] = f"infrastructure/{self._determine_infra_type(source)}"

        # Libs layer mappings
        for source in map_config.get("move_to_libs", []):
            if source.endswith("/"):
                source = source[:-1]
            migration_map[source] = f"libs/{self._determine_lib_type(source)}"

        # Integrations layer mappings
        for source in map_config.get("move_to_integrations", []):
            if source.endswith("/"):
                source = source[:-1]
            migration_map[source] = f"integrations/{self._determine_integration_type(source)}"

        return migration_map

    def _determine_app_type(self, source: str) -> str:
        """Determine app subdirectory based on source"""
        if "ui" in source.lower() or "web" in source.lower():
            return "web/shared"
        elif "education" in source.lower() or "mobile" in source.lower():
            return "mobile/shared"
        elif "cli" in source.lower():
            return "cli"
        else:
            return "web/shared"

    def _determine_core_type(self, source: str) -> str:
        """Determine core subdirectory based on source"""
        if "agents" in source.lower():
            return "agents"
        elif "agent_forge" in source.lower() or "forge" in source.lower():
            return "agent-forge"
        elif "rag" in source.lower():
            return "rag"
        elif "tokenomics" in source.lower():
            return "domain/tokenomics"
        else:
            return "domain"

    def _determine_infra_type(self, source: str) -> str:
        """Determine infrastructure subdirectory based on source"""
        if "p2p" in source.lower():
            return "p2p"
        elif "fog" in source.lower():
            return "fog"
        elif "edge" in source.lower():
            return "fog/edge"
        elif "monitoring" in source.lower():
            return "monitoring"
        else:
            return "data"

    def _determine_lib_type(self, source: str) -> str:
        """Determine libs subdirectory based on source"""
        if "hrrm" in source.lower() or "ml" in source.lower():
            return "ml-utils"
        elif "mobile" in source.lower():
            return "common"
        elif "crypto" in source.lower():
            return "crypto"
        else:
            return "common"

    def _determine_integration_type(self, source: str) -> str:
        """Determine integration subdirectory based on source"""
        if "automation" in source.lower():
            return "external-apis"
        elif "blockchain" in source.lower():
            return "blockchain"
        else:
            return "cloud-services"

    def run_migration(self, phase: int) -> None:
        """Run specific migration phase"""
        logger.info(f"Starting migration phase {phase}")

        if phase == 1:
            self._phase1_create_structure()
        elif phase == 2:
            self._phase2_move_clear_components()
        elif phase == 3:
            self._phase3_consolidate_duplicates()
        elif phase == 4:
            self._phase4_update_imports()
        elif phase == 5:
            self._phase5_cleanup()
        else:
            raise ValueError(f"Invalid phase: {phase}")

        logger.info(f"Completed migration phase {phase}")

    def _phase1_create_structure(self) -> None:
        """Phase 1: Create clean architecture directory structure"""
        logger.info("Creating clean architecture directory structure...")

        structure = self.config["directory_structure"]

        for layer, layer_structure in structure.items():
            self._create_layer_structure(layer, layer_structure)

        # Create __init__.py files
        self._create_init_files()

        # Create README files
        self._create_readme_files()

    def _create_layer_structure(self, layer: str, structure) -> None:
        """Create directory structure for a layer"""
        layer_path = self.project_root / layer

        if not self.dry_run:
            layer_path.mkdir(exist_ok=True)
        logger.info(f"Created layer directory: {layer_path}")

        if isinstance(structure, dict):
            for subdir, substructure in structure.items():
                subdir_path = layer_path / subdir
                if not self.dry_run:
                    subdir_path.mkdir(exist_ok=True)
                logger.info(f"Created subdirectory: {subdir_path}")

                if isinstance(substructure, list):
                    for subsubdir in substructure:
                        if subsubdir.endswith("/"):
                            subsubdir = subsubdir[:-1]
                        subsubdir_path = subdir_path / subsubdir
                        if not self.dry_run:
                            subsubdir_path.mkdir(exist_ok=True)
                        logger.info(f"Created subdirectory: {subsubdir_path}")
        elif isinstance(structure, list):
            for subdir in structure:
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                subdir_path = layer_path / subdir
                if not self.dry_run:
                    subdir_path.mkdir(exist_ok=True)
                logger.info(f"Created subdirectory: {subdir_path}")

    def _create_init_files(self) -> None:
        """Create __init__.py files in all directories"""
        for layer in ["apps", "core", "infrastructure", "devops", "libs", "integrations"]:
            layer_path = self.project_root / layer
            if layer_path.exists():
                for dir_path in layer_path.rglob("*"):
                    if dir_path.is_dir():
                        init_file = dir_path / "__init__.py"
                        if not init_file.exists() and not self.dry_run:
                            init_file.write_text(f'"""{"_".join(dir_path.parts[-2:])} module"""\n')
                        logger.debug(f"Created init file: {init_file}")

    def _create_readme_files(self) -> None:
        """Create README files for major directories"""
        readme_content = {
            "apps": "# Applications Layer\n\nUser interfaces and client applications.\n",
            "core": "# Core Layer\n\nBusiness logic and domain models.\n",
            "infrastructure": "# Infrastructure Layer\n\nTechnical implementation details.\n",
            "devops": "# DevOps Layer\n\nDevelopment operations and automation.\n",
            "libs": "# Libraries Layer\n\nShared libraries and utilities.\n",
            "integrations": "# Integrations Layer\n\nExternal system integrations.\n",
        }

        for layer, content in readme_content.items():
            readme_path = self.project_root / layer / "README.md"
            if not readme_path.exists() and not self.dry_run:
                readme_path.write_text(content)
            logger.info(f"Created README: {readme_path}")

    def _phase2_move_clear_components(self) -> None:
        """Phase 2: Move clearly identifiable components"""
        logger.info("Moving clear components to new structure...")

        for source, destination in self.migration_map.items():
            source_path = self.project_root / source
            dest_path = self.project_root / destination

            if source_path.exists():
                logger.info(f"Moving {source_path} -> {dest_path}")

                if not self.dry_run:
                    # Ensure destination directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Move the directory
                    shutil.move(str(source_path), str(dest_path))

                    # Track for rollback
                    self.moved_files.append((str(source_path), str(dest_path)))

                # Update imports in moved files
                self._update_imports_in_directory(dest_path)
            else:
                logger.warning(f"Source path not found: {source_path}")

    def _phase3_consolidate_duplicates(self) -> None:
        """Phase 3: Consolidate duplicate implementations"""
        logger.info("Consolidating duplicate implementations...")

        duplicates = self._find_duplicate_implementations()

        for duplicate_group in duplicates:
            self._consolidate_duplicate_group(duplicate_group)

    def _find_duplicate_implementations(self) -> list[list[Path]]:
        """Find duplicate implementations across the codebase"""
        duplicates = []

        # Simple heuristic: find files with similar names
        file_groups = {}

        for py_file in self.project_root.rglob("*.py"):
            if "test" in py_file.name or "__pycache__" in str(py_file):
                continue

            base_name = py_file.stem
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(py_file)

        # Filter to groups with multiple files
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Check if they're actually similar implementations
                if self._are_similar_implementations(files):
                    duplicates.append(files)

        return duplicates

    def _are_similar_implementations(self, files: list[Path]) -> bool:
        """Check if files contain similar implementations"""
        try:
            contents = []
            for file_path in files:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    contents.append(content)

            # Simple similarity check: compare class and function names
            class_function_sets = []
            for content in contents:
                try:
                    tree = ast.parse(content)
                    names = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef | ast.FunctionDef):
                            names.add(node.name)
                    class_function_sets.append(names)
                except:
                    return False

            # If any two sets have >50% overlap, consider them similar
            for i, set1 in enumerate(class_function_sets):
                for j, set2 in enumerate(class_function_sets[i + 1 :], i + 1):
                    if set1 and set2:
                        overlap = len(set1.intersection(set2))
                        total = len(set1.union(set2))
                        if overlap / total > 0.5:
                            return True

            return False
        except Exception as e:
            logger.warning(f"Error checking similarity: {e}")
            return False

    def _consolidate_duplicate_group(self, files: list[Path]) -> None:
        """Consolidate a group of duplicate files"""
        logger.info(f"Consolidating duplicates: {[str(f) for f in files]}")

        # Choose the best implementation (heuristic: newest, largest, or in core/)
        best_file = self._choose_best_implementation(files)

        logger.info(f"Choosing {best_file} as canonical implementation")

        if not self.dry_run:
            # Remove other implementations
            for file_path in files:
                if file_path != best_file:
                    logger.info(f"Removing duplicate: {file_path}")
                    file_path.unlink()

    def _choose_best_implementation(self, files: list[Path]) -> Path:
        """Choose the best implementation from duplicates"""
        # Prioritize files in core/ layer
        core_files = [f for f in files if "core/" in str(f)]
        if core_files:
            return max(core_files, key=lambda f: f.stat().st_size)

        # Otherwise choose largest file
        return max(files, key=lambda f: f.stat().st_size)

    def _phase4_update_imports(self) -> None:
        """Phase 4: Update all import statements"""
        logger.info("Updating import statements...")

        # Build import mapping
        import_map = self._build_import_map()

        # Update imports in all Python files
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            self._update_imports_in_file(py_file, import_map)

    def _build_import_map(self) -> dict[str, str]:
        """Build mapping of old imports to new imports"""
        import_map = {}

        for old_path, new_path in self.migration_map.items():
            # Convert file paths to import paths
            old_import = old_path.replace("/", ".").replace("packages.", "")
            new_import = new_path.replace("/", ".")

            import_map[old_import] = new_import

        return import_map

    def _update_imports_in_file(self, file_path: Path, import_map: dict[str, str]) -> None:
        """Update imports in a single file"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Update imports using import map
            for old_import, new_import in import_map.items():
                content = content.replace(f"from {old_import}", f"from {new_import}")
                content = content.replace(f"import {old_import}", f"import {new_import}")

            # Write back if changed
            if content != original_content and not self.dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Updated imports in: {file_path}")

        except Exception as e:
            logger.warning(f"Error updating imports in {file_path}: {e}")

    def _update_imports_in_directory(self, directory: Path) -> None:
        """Update imports in all files in a directory"""
        if not directory.exists():
            return

        for py_file in directory.rglob("*.py"):
            self._update_imports_in_file(py_file, self._build_import_map())

    def _phase5_cleanup(self) -> None:
        """Phase 5: Clean up old structures"""
        logger.info("Cleaning up old structures...")

        # Remove empty directories
        self._remove_empty_directories()

        # Remove deprecated folders
        deprecated_folders = ["deprecated", "archive", "backup*", "old_*"]

        for pattern in deprecated_folders:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    logger.info(f"Removing deprecated folder: {path}")
                    if not self.dry_run:
                        shutil.rmtree(path)

        # Clean up packages directory if empty
        packages_dir = self.project_root / "packages"
        if packages_dir.exists() and not any(packages_dir.iterdir()):
            logger.info(f"Removing empty packages directory: {packages_dir}")
            if not self.dry_run:
                packages_dir.rmdir()

    def _remove_empty_directories(self) -> None:
        """Remove empty directories"""
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                logger.info(f"Removing empty directory: {dir_path}")
                if not self.dry_run:
                    dir_path.rmdir()

    def validate_migration(self) -> list[str]:
        """Validate the migration results"""
        errors = []

        # Check layer boundary violations
        layer_violations = self._check_layer_boundaries()
        errors.extend(layer_violations)

        # Check for broken imports
        import_errors = self._check_imports()
        errors.extend(import_errors)

        # Check for missing files
        missing_files = self._check_missing_files()
        errors.extend(missing_files)

        return errors

    def _check_layer_boundaries(self) -> list[str]:
        """Check for layer boundary violations"""
        violations = []

        self.config["architecture"]["layers"]

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            layer = self._determine_file_layer(py_file)
            if not layer:
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Parse imports
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import | ast.ImportFrom):
                        import_path = self._get_import_path(node)
                        if import_path and self._violates_layer_rules(layer, import_path):
                            violations.append(f"Layer violation in {py_file}: {layer} imports {import_path}")
            except Exception as e:
                logger.warning(f"Error checking {py_file}: {e}")

        return violations

    def _determine_file_layer(self, file_path: Path) -> str:
        """Determine which layer a file belongs to"""
        parts = file_path.parts

        for part in parts:
            if part in ["apps", "core", "infrastructure", "devops", "libs", "integrations"]:
                return part

        return None

    def _get_import_path(self, node) -> str:
        """Extract import path from AST node"""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            return node.module if node.module else None
        return None

    def _violates_layer_rules(self, layer: str, import_path: str) -> bool:
        """Check if import violates layer rules"""
        layer_config = self.config["architecture"]["layers"].get(layer, {})
        forbidden = layer_config.get("forbidden_dependencies", [])

        for forbidden_dep in forbidden:
            if import_path.startswith(forbidden_dep):
                return True

        return False

    def _check_imports(self) -> list[str]:
        """Check for broken imports"""
        errors = []

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Try to compile to check for syntax errors
                ast.parse(content)

                # TODO: Add more sophisticated import checking

            except SyntaxError as e:
                errors.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                errors.append(f"Error checking {py_file}: {e}")

        return errors

    def _check_missing_files(self) -> list[str]:
        """Check for missing critical files"""
        missing = []

        critical_files = ["core/__init__.py", "infrastructure/__init__.py", "apps/__init__.py", "libs/__init__.py"]

        for file_path in critical_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing.append(f"Missing critical file: {file_path}")

        return missing

    def rollback(self) -> None:
        """Rollback migration changes"""
        logger.info("Rolling back migration changes...")

        for dest_path, source_path in reversed(self.moved_files):
            if Path(dest_path).exists():
                logger.info(f"Rolling back: {dest_path} -> {source_path}")
                if not self.dry_run:
                    Path(source_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(dest_path, source_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate AIVillage to clean architecture")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], required=True, help="Migration phase to run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--validate", action="store_true", help="Validate migration results")
    parser.add_argument("--rollback", action="store_true", help="Rollback previous migration")

    args = parser.parse_args()

    migrator = CleanArchitectureMigrator(PROJECT_ROOT, dry_run=args.dry_run)

    try:
        if args.rollback:
            migrator.rollback()
        elif args.validate:
            errors = migrator.validate_migration()
            if errors:
                logger.error("Migration validation failed:")
                for error in errors:
                    logger.error(f"  {error}")
                sys.exit(1)
            else:
                logger.info("Migration validation passed!")
        else:
            migrator.run_migration(args.phase)

            # Auto-validate after migration
            errors = migrator.validate_migration()
            if errors:
                logger.warning("Migration completed with warnings:")
                for error in errors:
                    logger.warning(f"  {error}")
            else:
                logger.info("Migration completed successfully!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
