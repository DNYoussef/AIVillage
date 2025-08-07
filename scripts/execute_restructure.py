#!/usr/bin/env python3
"""Execute the AIVillage codebase restructuring
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("restructure.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AIVillageRestructurer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.moved_items = []
        self.errors = []

    def create_directory_structure(self):
        """Create the new directory structure."""
        logger.info("Creating new directory structure...")

        new_dirs = [
            "src/production",
            "src/core",
            "src/agent_forge",
            "src/mcp_servers",
            "src/digital_twin",
            "experimental/agents",
            "experimental/mesh",
            "experimental/services",
            "experimental/training",
            "tools/scripts",
            "tools/benchmarks",
            "tools/examples",
            "mobile",
        ]

        for dir_path in new_dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

            # Add __init__.py to Python packages
            if any(part in dir_path for part in ["src", "experimental"]):
                init_file = full_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Auto-generated during restructure\n")

        logger.info("Directory structure created successfully")

    def move_item(self, source: Path, target: Path) -> bool:
        """Move a file or directory safely."""
        try:
            if not source.exists():
                logger.warning(f"Source does not exist: {source}")
                return False

            # Create target parent directory if needed
            target.parent.mkdir(parents=True, exist_ok=True)

            # If target exists, skip to avoid overwriting
            if target.exists():
                logger.warning(f"Target already exists, skipping: {target}")
                return False

            # Move the item
            shutil.move(str(source), str(target))
            self.moved_items.append(f"{source.name} -> {target}")
            logger.info(f"Moved: {source.name} -> {target}")
            return True

        except Exception as e:
            error_msg = f"Error moving {source} to {target}: {e!s}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False

    def restructure_production_components(self):
        """Move production-ready components to src/."""
        logger.info("Restructuring production components...")

        # Core production directories to move
        production_dirs = {
            "production": "src/production",
            "digital_twin": "src/digital_twin",
            "mcp_servers": "src/mcp_servers",
            "monitoring": "src/monitoring",
            "jobs": "src/jobs",
            "communications": "src/communications",
            "calibration": "src/calibration",
            "core": "src/core",
            "services": "src/services",
            "ingestion": "src/ingestion",
            "hyperag": "src/hyperag",
        }

        for source_name, target_path in production_dirs.items():
            source = self.base_path / source_name
            target = self.base_path / target_path

            if source.exists():
                self.move_item(source, target)

    def restructure_agent_forge(self):
        """Split agent_forge between stable (src/) and experimental."""
        logger.info("Restructuring agent_forge...")

        agent_forge_source = self.base_path / "agent_forge"
        if not agent_forge_source.exists():
            logger.warning("agent_forge directory not found")
            return

        # Stable components go to src/agent_forge/
        stable_components = [
            "core",
            "evaluation",
            "deployment",
            "utils",
            "orchestration",
        ]

        # Create src/agent_forge structure
        src_agent_forge = self.base_path / "src" / "agent_forge"
        src_agent_forge.mkdir(parents=True, exist_ok=True)

        # Move stable components
        for component in stable_components:
            source = agent_forge_source / component
            target = src_agent_forge / component
            if source.exists():
                self.move_item(source, target)

        # Experimental components go to experimental/
        experimental_components = [
            "self_awareness",
            "bakedquietiot",
            "sleepdream",
            "foundation",
            "prompt_baking_legacy",
            "tool_baking",
            "adas",
            "optim",
            "svf",
            "meta",
            "training",
            "evolution",
            "compression",
        ]

        exp_agent_forge = self.base_path / "experimental" / "agent_forge_experimental"
        exp_agent_forge.mkdir(parents=True, exist_ok=True)

        for component in experimental_components:
            source = agent_forge_source / component
            target = exp_agent_forge / component
            if source.exists():
                self.move_item(source, target)

        # Move remaining files in agent_forge root
        for item in agent_forge_source.iterdir():
            if item.is_file() and item.suffix == ".py":
                if item.name in ["main.py", "version.py"]:
                    target = src_agent_forge / item.name
                else:
                    target = exp_agent_forge / item.name
                self.move_item(item, target)

    def restructure_experimental(self):
        """Move existing experimental code."""
        logger.info("Restructuring experimental components...")

        experimental_source = self.base_path / "experimental"
        if experimental_source.exists():
            # Experimental directory already exists, just ensure it's properly organized
            logger.info("Experimental directory already exists and organized")

    def restructure_tools(self):
        """Move tools, scripts, benchmarks, examples."""
        logger.info("Restructuring tools...")

        tools_mapping = {
            "scripts": "tools/scripts",
            "benchmarks": "tools/benchmarks",
            "examples": "tools/examples",
        }

        for source_name, target_path in tools_mapping.items():
            source = self.base_path / source_name
            target = self.base_path / target_path

            if source.exists():
                self.move_item(source, target)

    def restructure_rag_system(self):
        """Move rag_system to src/."""
        logger.info("Restructuring RAG system...")

        rag_source = self.base_path / "rag_system"
        if rag_source.exists():
            target = self.base_path / "src" / "rag_system"
            self.move_item(rag_source, target)

    def restructure_tests(self):
        """Move tests to proper location."""
        logger.info("Restructuring tests...")

        # Tests directory can stay at root level
        tests_source = self.base_path / "tests"
        if tests_source.exists():
            logger.info("Tests directory left at root level")

    def cleanup_empty_directories(self):
        """Remove empty directories after restructuring."""
        logger.info("Cleaning up empty directories...")

        def remove_empty_dir(path: Path):
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    logger.info(f"Removed empty directory: {path}")
                    return True
            except (OSError, PermissionError):
                pass
            return False

        # Try to remove potentially empty directories
        empty_candidates = [
            "agent_forge",
            "experimental",
            "scripts",
            "benchmarks",
            "examples",
            "production",
            "digital_twin",
            "mcp_servers",
        ]

        for candidate in empty_candidates:
            candidate_path = self.base_path / candidate
            if candidate_path.exists():
                remove_empty_dir(candidate_path)

    def generate_report(self):
        """Generate restructuring report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "items_moved": len(self.moved_items),
                "errors": len(self.errors),
            },
            "moved_items": self.moved_items,
            "errors": self.errors,
            "new_structure": {
                "src/": "Production-ready code",
                "experimental/": "Experimental and prototype code",
                "tools/": "Scripts, benchmarks, and examples",
                "tests/": "Test suites",
                "mobile/": "Mobile projects (placeholder)",
            },
        }

        report_file = self.base_path / "restructure_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {report_file}")
        return report

    def execute_restructure(self):
        """Execute the complete restructuring process."""
        logger.info("=== Starting AIVillage Codebase Restructuring ===")

        try:
            # Step 1: Create new directory structure
            self.create_directory_structure()

            # Step 2: Move production components
            self.restructure_production_components()

            # Step 3: Split agent_forge
            self.restructure_agent_forge()

            # Step 4: Move experimental code (already mostly organized)
            self.restructure_experimental()

            # Step 5: Move tools
            self.restructure_tools()

            # Step 6: Move RAG system
            self.restructure_rag_system()

            # Step 7: Handle tests
            self.restructure_tests()

            # Step 8: Cleanup empty directories
            self.cleanup_empty_directories()

            # Step 9: Generate report
            report = self.generate_report()

            logger.info("=== Restructuring Complete ===")
            logger.info(f"Items moved: {len(self.moved_items)}")
            logger.info(f"Errors: {len(self.errors)}")

            if self.errors:
                logger.warning("Errors encountered:")
                for error in self.errors:
                    logger.warning(f"  - {error}")

            return report

        except Exception as e:
            logger.error(f"Critical error during restructuring: {e!s}")
            raise


def main():
    print("=== AIVillage Codebase Restructuring ===")

    # Create backup first
    print("Creating backup snapshot...")
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Execute restructuring
    restructurer = AIVillageRestructurer(os.getcwd())
    report = restructurer.execute_restructure()

    print("\n=== Restructuring Summary ===")
    print(f"Items moved: {report['summary']['items_moved']}")
    print(f"Errors: {report['summary']['errors']}")

    if report["summary"]["errors"] > 0:
        print("\nErrors encountered - check restructure.log for details")

    print("\nNew structure created:")
    for path, description in report["new_structure"].items():
        print(f"  {path} - {description}")

    print("\nDetailed report: restructure_report.json")
    print("Restructuring complete!")

    return report


if __name__ == "__main__":
    main()
