#!/usr/bin/env python3
"""AIVillage Codebase Restructuring Script.

This script reorganizes the codebase into a clean, production-ready structure:
- src/ for production-ready code
- experimental/ for experimental/prototype code
- tools/ for scripts, benchmarks, and examples
- mobile/ for mobile projects

The script performs the restructuring safely with validation checks.
"""

import json
import logging
import os
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodebaseRestructurer:
    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        self.moved_files = []
        self.errors = []

        # Define the new structure mapping
        self.structure_map = {
            # Production-ready code -> src/
            "production": "src/production",
            "digital_twin": "src/digital_twin",
            "mcp_servers": "src/mcp_servers",
            # Core infrastructure (stable parts of agent_forge)
            "agent_forge/core": "src/agent_forge/core",
            "agent_forge/evaluation": "src/agent_forge/evaluation",
            # Experimental code -> experimental/
            "experimental": "experimental",
            "agent_forge/self_awareness": "experimental/agent_forge_experimental/self_awareness",
            "agent_forge/bakedquietiot": "experimental/agent_forge_experimental/bakedquietiot",
            # Tools consolidation -> tools/
            "scripts": "tools/scripts",
            "benchmarks": "tools/benchmarks",
            "examples": "tools/examples",
            # Mobile projects -> mobile/ (if maintained)
            # Will be handled separately based on maintenance status
        }

        # Files/directories to skip during restructuring
        self.skip_items = {
            ".git",
            ".github",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            "new_env",
            ".env",
            "venv",
            "env",
            # Config and root files stay at root
            "pyproject.toml",
            "requirements*.txt",
            "setup.py",
            "Dockerfile*",
            "docker-compose*.yml",
            "Makefile",
            "pytest.ini",
            "*.md",
            "*.txt",
            "LICENSE",
            "CHANGELOG.md",
            "README.md",
            "*.log",
            "*.json",
            "*.xml",
            "coverage.xml",
            "*.db",
            "*.sh",
            "*.py",
            "main.py",
            "server.py",
        }

    def validate_source_exists(self, source_path: Path) -> bool:
        """Validate that source path exists and is accessible."""
        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return False
        return True

    def create_target_structure(self) -> None:
        """Create the new directory structure."""
        logger.info("Creating new directory structure...")

        directories_to_create = [
            "src/production",
            "src/core",
            "src/agent_forge",
            "src/mcp_servers",
            "src/digital_twin",
            "src/rag_system",
            "experimental/agents",
            "experimental/mesh",
            "experimental/services",
            "experimental/training",
            "experimental/agent_forge_experimental",
            "tools/scripts",
            "tools/benchmarks",
            "tools/examples",
            "mobile/android-sdk",
            "mobile/mobile-app",
            "mobile/monorepo",
        ]

        for dir_path in directories_to_create:
            target_dir = self.base_path / dir_path
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {target_dir}")

            # Create __init__.py files for Python packages
            if "src/" in dir_path or "experimental/" in dir_path:
                init_file = target_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Auto-generated during restructure\n")

    def should_skip_item(self, item_path: Path) -> bool:
        """Check if an item should be skipped during restructuring."""
        item_name = item_path.name

        # Skip items in skip list
        for skip_pattern in self.skip_items:
            if "*" in skip_pattern:
                # Handle wildcard patterns
                pattern = skip_pattern.replace("*", "")
                if pattern in item_name:
                    return True
            elif item_name == skip_pattern:
                return True

        # Skip if it's already in the new structure
        return item_name in ["src", "experimental", "tools", "mobile"]

    def move_directory_contents(self, source_dir: Path, target_dir: Path) -> None:
        """Move contents of source directory to target directory."""
        if not self.validate_source_exists(source_dir):
            return

        logger.info(f"Moving {source_dir} -> {target_dir}")

        try:
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)

            # Move all contents from source to target
            for item in source_dir.iterdir():
                if self.should_skip_item(item):
                    continue

                target_item = target_dir / item.name

                if target_item.exists():
                    logger.warning(f"Target already exists, skipping: {target_item}")
                    continue

                shutil.move(str(item), str(target_item))
                self.moved_files.append(f"{item} -> {target_item}")
                logger.info(f"Moved: {item.name}")

        except Exception as e:
            error_msg = f"Error moving {source_dir} to {target_dir}: {e!s}"
            logger.exception(error_msg)
            self.errors.append(error_msg)

    def restructure_production_components(self) -> None:
        """Move production-ready components to src/."""
        logger.info("Restructuring production components...")

        # Move production directory - preserve internal structure
        production_source = self.base_path / "production"
        if production_source.exists():
            self.move_directory_contents(
                production_source, self.base_path / "src" / "production"
            )

        # Move digital_twin
        digital_twin_source = self.base_path / "digital_twin"
        if digital_twin_source.exists():
            self.move_directory_contents(
                digital_twin_source, self.base_path / "src" / "digital_twin"
            )

        # Move mcp_servers
        mcp_source = self.base_path / "mcp_servers"
        if mcp_source.exists():
            self.move_directory_contents(
                mcp_source, self.base_path / "src" / "mcp_servers"
            )

        # Move other production-ready directories
        other_production_dirs = ["monitoring", "jobs", "communications", "calibration"]
        for prod_dir in other_production_dirs:
            prod_source = self.base_path / prod_dir
            if prod_source.exists():
                self.move_directory_contents(
                    prod_source, self.base_path / "src" / prod_dir
                )

    def restructure_agent_forge(self) -> None:
        """Split agent_forge between src/ (stable) and experimental/."""
        logger.info("Restructuring agent_forge...")

        agent_forge_source = self.base_path / "agent_forge"
        if not agent_forge_source.exists():
            return

        # Stable components go to src/agent_forge/
        stable_components = ["core", "evaluation"]

        for component in stable_components:
            component_path = agent_forge_source / component
            if component_path.exists():
                self.move_directory_contents(
                    component_path, self.base_path / "src" / "agent_forge" / component
                )

        # Experimental components go to experimental/agent_forge_experimental/
        experimental_components = ["self_awareness", "bakedquietiot"]

        for component in experimental_components:
            component_path = agent_forge_source / component
            if component_path.exists():
                self.move_directory_contents(
                    component_path,
                    self.base_path
                    / "experimental"
                    / "agent_forge_experimental"
                    / component,
                )

    def restructure_experimental(self) -> None:
        """Move experimental code to experimental/."""
        logger.info("Restructuring experimental components...")

        experimental_source = self.base_path / "experimental"
        if experimental_source.exists():
            # Move existing experimental structure as-is
            for item in experimental_source.iterdir():
                if self.should_skip_item(item):
                    continue
                target_path = self.base_path / "experimental" / item.name
                if not target_path.exists() and item.is_dir():
                    self.move_directory_contents(item, target_path)

    def restructure_tools(self) -> None:
        """Consolidate tools, scripts, benchmarks, and examples."""
        logger.info("Restructuring tools...")

        # Move scripts
        scripts_source = self.base_path / "scripts"
        if scripts_source.exists():
            self.move_directory_contents(
                scripts_source, self.base_path / "tools" / "scripts"
            )

        # Move benchmarks
        benchmarks_source = self.base_path / "benchmarks"
        if benchmarks_source.exists():
            self.move_directory_contents(
                benchmarks_source, self.base_path / "tools" / "benchmarks"
            )

        # Move examples
        examples_source = self.base_path / "examples"
        if examples_source.exists():
            self.move_directory_contents(
                examples_source, self.base_path / "tools" / "examples"
            )

    def handle_mobile_projects(self) -> None:
        """Handle mobile projects - move if maintained, create submodule refs if separate."""
        logger.info("Handling mobile projects...")

        # For now, create placeholder structure
        # In real implementation, this would check maintenance status
        mobile_dirs = ["android-sdk", "mobile-app", "monorepo"]

        for mobile_dir in mobile_dirs:
            mobile_path = self.base_path / mobile_dir
            if mobile_path.exists():
                self.move_directory_contents(
                    mobile_path, self.base_path / "mobile" / mobile_dir
                )

    def cleanup_empty_directories(self) -> None:
        """Remove empty directories after restructuring."""
        logger.info("Cleaning up empty directories...")

        def remove_empty_dirs(path: Path) -> None:
            """Recursively remove empty directories."""
            if not path.is_dir():
                return

            # Remove empty subdirectories first
            for item in path.iterdir():
                if item.is_dir():
                    remove_empty_dirs(item)

            # Remove this directory if it's empty
            try:
                if not any(path.iterdir()):
                    path.rmdir()
                    logger.info(f"Removed empty directory: {path}")
            except OSError:
                pass  # Directory not empty or permission issue

        # Clean up potential empty directories
        cleanup_candidates = [
            "production",
            "digital_twin",
            "mcp_servers",
            "agent_forge",
            "experimental",
            "scripts",
            "benchmarks",
            "examples",
        ]

        for candidate in cleanup_candidates:
            candidate_path = self.base_path / candidate
            if candidate_path.exists():
                remove_empty_dirs(candidate_path)

    def update_import_paths(self) -> None:
        """Update import paths in Python files to reflect new structure."""
        logger.info("Updating import paths...")

        # This is a simplified version - full implementation would need more sophisticated
        # import path detection and replacement
        import_replacements = {
            "from production.": "from src.production.",
            "import production.": "import src.production.",
            "from digital_twin.": "from src.digital_twin.",
            "import digital_twin.": "import src.digital_twin.",
            "from mcp_servers.": "from src.mcp_servers.",
            "import mcp_servers.": "import src.mcp_servers.",
            "from agent_forge.core.": "from src.agent_forge.core.",
            "from agent_forge.evaluation.": "from src.agent_forge.evaluation.",
            "from scripts.": "from tools.scripts.",
            "import scripts.": "import tools.scripts.",
        }

        # Find all Python files in new structure
        python_files = list(self.base_path.rglob("*.py"))

        for py_file in python_files:
            if self.should_skip_item(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                modified = False

                for old_import, new_import in import_replacements.items():
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        modified = True

                if modified:
                    py_file.write_text(content, encoding="utf-8")
                    logger.info(f"Updated imports in: {py_file}")

            except Exception as e:
                logger.warning(f"Could not update imports in {py_file}: {e!s}")

    def generate_restructure_report(self):
        """Generate a report of the restructuring process."""
        report = {
            "restructure_summary": {
                "timestamp": str(Path().resolve()),
                "files_moved": len(self.moved_files),
                "errors_encountered": len(self.errors),
            },
            "moved_files": self.moved_files,
            "errors": self.errors,
            "new_structure": {
                "src/": "Production-ready code",
                "experimental/": "Experimental and prototype code",
                "tools/": "Scripts, benchmarks, and examples",
                "mobile/": "Mobile projects",
            },
        }

        report_file = self.base_path / "restructure_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Restructure report saved to: {report_file}")
        return report

    def run_restructure(self):
        """Execute the complete restructuring process."""
        logger.info("Starting codebase restructuring...")

        try:
            # Step 1: Create new directory structure
            self.create_target_structure()

            # Step 2: Move production components
            self.restructure_production_components()

            # Step 3: Split agent_forge
            self.restructure_agent_forge()

            # Step 4: Move experimental code
            self.restructure_experimental()

            # Step 5: Consolidate tools
            self.restructure_tools()

            # Step 6: Handle mobile projects
            self.handle_mobile_projects()

            # Step 7: Clean up empty directories
            self.cleanup_empty_directories()

            # Step 8: Update import paths
            self.update_import_paths()

            # Step 9: Generate report
            report = self.generate_restructure_report()

            logger.info("Codebase restructuring completed successfully!")
            logger.info(f"Files moved: {len(self.moved_files)}")
            logger.info(f"Errors: {len(self.errors)}")

            return report

        except Exception as e:
            logger.exception(f"Critical error during restructuring: {e!s}")
            raise


if __name__ == "__main__":
    base_path = os.getcwd()  # Current working directory
    restructurer = CodebaseRestructurer(base_path)
    restructurer.run_restructure()
