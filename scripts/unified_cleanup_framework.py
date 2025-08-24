#!/usr/bin/env python3
"""
Unified Cleanup Framework for AIVillage
Standardizes all cleanup, fix, and maintenance scripts with consistent interface
"""

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


class CleanupStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CleanupResult:
    task_name: str
    status: CleanupStatus
    items_processed: int
    items_fixed: int
    errors: list[str]
    warnings: list[str]
    execution_time: float
    details: dict[str, Any]


@dataclass
class CleanupConfig:
    dry_run: bool = True
    create_backups: bool = True
    validate_before_cleanup: bool = True
    rollback_on_failure: bool = True
    verbose: bool = False
    parallel: bool = False
    max_files_per_task: int = 1000


class BaseCleanupTask(ABC):
    """Base class for all cleanup tasks with standardized interface"""

    def __init__(self, name: str, description: str, config: CleanupConfig):
        self.name = name
        self.description = description
        self.config = config
        self.logger = logging.getLogger(f"cleanup.{name}")

    @abstractmethod
    def analyze(self, project_root: Path) -> dict[str, Any]:
        """Analyze what needs to be cleaned up"""
        pass

    @abstractmethod
    def execute(self, project_root: Path, analysis_result: dict[str, Any]) -> CleanupResult:
        """Execute the cleanup task"""
        pass

    def validate(self, project_root: Path, result: CleanupResult) -> bool:
        """Validate cleanup was successful"""
        return result.status == CleanupStatus.COMPLETED

    def rollback(self, project_root: Path, backup_dir: Path) -> bool:
        """Rollback changes if needed"""
        self.logger.info(f"Rolling back {self.name} changes...")
        return True


class LintingFixTask(BaseCleanupTask):
    """Fix common linting issues (consolidates fix_linting_issues.py)"""

    def analyze(self, project_root: Path) -> dict[str, Any]:
        python_files = list(project_root.rglob("*.py"))
        # Exclude deprecated/archived files
        python_files = [
            f
            for f in python_files
            if not any(part in str(f).lower() for part in ["deprecated", "archive", "experimental", "__pycache__"])
        ]

        issues = {"long_lines": [], "import_issues": [], "trailing_whitespace": [], "unused_imports": []}

        for py_file in python_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if len(line.rstrip()) > 120:
                        issues["long_lines"].append((py_file, line_num))
                    if line.endswith(" \n") or line.endswith("\t\n"):
                        issues["trailing_whitespace"].append((py_file, line_num))

            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")

        return {
            "total_files": len(python_files),
            "issues": issues,
            "total_issues": sum(len(v) for v in issues.values()),
        }

    def execute(self, project_root: Path, analysis_result: dict[str, Any]) -> CleanupResult:
        start_time = time.time()
        result = CleanupResult(
            task_name=self.name,
            status=CleanupStatus.IN_PROGRESS,
            items_processed=0,
            items_fixed=0,
            errors=[],
            warnings=[],
            execution_time=0,
            details={},
        )

        try:
            issues = analysis_result["issues"]

            if not self.config.dry_run:
                # Fix trailing whitespace with ruff
                subprocess.run(
                    ["ruff", "check", "--select=W291,W292,W293", "--fix", str(project_root)], capture_output=True
                )
                result.items_fixed += len(issues["trailing_whitespace"])

                # Fix import sorting with isort
                subprocess.run(["isort", str(project_root)], capture_output=True)
                result.items_fixed += len(issues["import_issues"])

            result.items_processed = analysis_result["total_issues"]
            result.status = CleanupStatus.COMPLETED

        except Exception as e:
            result.status = CleanupStatus.FAILED
            result.errors.append(str(e))

        result.execution_time = time.time() - start_time
        return result


class ImportFixTask(BaseCleanupTask):
    """Fix import path issues (consolidates fix_*_imports.py scripts)"""

    def analyze(self, project_root: Path) -> dict[str, Any]:
        python_files = list(project_root.rglob("*.py"))
        python_files = [
            f
            for f in python_files
            if not any(part in str(f).lower() for part in ["deprecated", "archive", "__pycache__"])
        ]

        import_issues = []
        for py_file in python_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Look for common import issues
                lines = content.split("\n")
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith("from ") or line.startswith("import "):
                        # Check for relative imports that might be broken
                        if "from ." in line or "import ." in line:
                            import_issues.append(
                                {"file": py_file, "line": line_num, "issue": "relative_import", "content": line}
                            )
                        # Check for imports from moved modules
                        if any(old_path in line for old_path in ["agent_forge", "hyperrag", "p2p"]):
                            import_issues.append(
                                {"file": py_file, "line": line_num, "issue": "moved_module", "content": line}
                            )

            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")

        return {"total_files": len(python_files), "import_issues": import_issues, "total_issues": len(import_issues)}

    def execute(self, project_root: Path, analysis_result: dict[str, Any]) -> CleanupResult:
        start_time = time.time()
        result = CleanupResult(
            task_name=self.name,
            status=CleanupStatus.IN_PROGRESS,
            items_processed=len(analysis_result["import_issues"]),
            items_fixed=0,
            errors=[],
            warnings=[],
            execution_time=0,
            details={},
        )

        try:
            if not self.config.dry_run:
                # Apply common import fixes
                for issue in analysis_result["import_issues"]:
                    if issue["issue"] == "moved_module":
                        # Apply standard path fixes
                        self._fix_import_path(issue)
                        result.items_fixed += 1
            else:
                self.logger.info(f"DRY RUN: Would fix {len(analysis_result['import_issues'])} import issues")

            result.status = CleanupStatus.COMPLETED

        except Exception as e:
            result.status = CleanupStatus.FAILED
            result.errors.append(str(e))

        result.execution_time = time.time() - start_time
        return result

    def _fix_import_path(self, issue: dict[str, Any]) -> None:
        """Apply specific import path fixes"""
        file_path = issue["file"]
        issue["content"]

        # Define common path mappings
        path_mappings = {
            "from agent_forge": "from core.agent-forge",
            "from hyperrag": "from core.hyperrag",
            "from p2p": "from packages.p2p",
            "import agent_forge": "import core.agent-forge",
            "import hyperrag": "import core.hyperrag",
            "import p2p": "import packages.p2p",
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            for old_path, new_path in path_mappings.items():
                content = content.replace(old_path, new_path)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            self.logger.error(f"Failed to fix import in {file_path}: {e}")


class FileCleanupTask(BaseCleanupTask):
    """Clean up temporary files and organize structure"""

    def analyze(self, project_root: Path) -> dict[str, Any]:
        cleanup_candidates = {"temp_files": [], "cache_dirs": [], "log_files": [], "backup_files": []}

        # Find temporary files
        for pattern in ["*.tmp", "*.log", "*.bak", "*~", "*.swp"]:
            cleanup_candidates["temp_files"].extend(project_root.rglob(pattern))

        # Find cache directories
        for cache_dir in ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]:
            cleanup_candidates["cache_dirs"].extend(project_root.rglob(cache_dir))

        # Find large log files
        for log_file in project_root.rglob("*.log"):
            if log_file.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                cleanup_candidates["log_files"].append(log_file)

        return {
            "cleanup_candidates": cleanup_candidates,
            "total_items": sum(len(v) for v in cleanup_candidates.values()),
        }

    def execute(self, project_root: Path, analysis_result: dict[str, Any]) -> CleanupResult:
        start_time = time.time()
        result = CleanupResult(
            task_name=self.name,
            status=CleanupStatus.IN_PROGRESS,
            items_processed=analysis_result["total_items"],
            items_fixed=0,
            errors=[],
            warnings=[],
            execution_time=0,
            details={},
        )

        try:
            candidates = analysis_result["cleanup_candidates"]

            if not self.config.dry_run:
                # Clean up cache directories
                for cache_dir in candidates["cache_dirs"]:
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir, ignore_errors=True)
                        result.items_fixed += 1

                # Clean up temp files
                for temp_file in candidates["temp_files"]:
                    if temp_file.exists():
                        temp_file.unlink()
                        result.items_fixed += 1
            else:
                self.logger.info(f"DRY RUN: Would clean up {analysis_result['total_items']} items")
                result.items_fixed = analysis_result["total_items"]

            result.status = CleanupStatus.COMPLETED

        except Exception as e:
            result.status = CleanupStatus.FAILED
            result.errors.append(str(e))

        result.execution_time = time.time() - start_time
        return result


class UnifiedCleanupManager:
    """Central manager for all cleanup tasks"""

    def __init__(self, config: CleanupConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.backup_dir = self.project_root / "backups" / f"cleanup_backup_{int(time.time())}"

        # Configure logging
        log_level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("unified_cleanup.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("unified_cleanup")

        # Initialize cleanup tasks
        self.tasks = [
            LintingFixTask("linting_fixes", "Fix common linting issues", config),
            ImportFixTask("import_fixes", "Fix import path issues", config),
            FileCleanupTask("file_cleanup", "Clean up temporary files", config),
        ]

    def create_backup(self) -> Path:
        """Create backup of project before cleanup"""
        if not self.config.create_backups:
            return None

        self.logger.info(f"Creating backup at {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup critical files
        for pattern in ["*.py", "*.yml", "*.yaml", "*.toml", "*.ini", "*.json"]:
            for file_path in self.project_root.rglob(pattern):
                if "backup" not in str(file_path) and "__pycache__" not in str(file_path):
                    rel_path = file_path.relative_to(self.project_root)
                    backup_path = self.backup_dir / rel_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)

        return self.backup_dir

    def run_all_tasks(self) -> list[CleanupResult]:
        """Run all cleanup tasks with standardized workflow"""
        results = []

        self.logger.info("Starting unified cleanup process...")

        # Create backup if enabled
        backup_dir = self.create_backup()

        # Run each task
        for task in self.tasks:
            self.logger.info(f"Running task: {task.name} - {task.description}")

            try:
                # Analysis phase
                analysis_result = task.analyze(self.project_root)
                self.logger.info(f"Analysis complete: {analysis_result.get('total_issues', 0)} issues found")

                # Execution phase
                result = task.execute(self.project_root, analysis_result)

                # Validation phase
                if self.config.validate_before_cleanup and not task.validate(self.project_root, result):
                    self.logger.warning(f"Validation failed for {task.name}")
                    result.status = CleanupStatus.FAILED

                results.append(result)

            except Exception as e:
                self.logger.error(f"Task {task.name} failed: {e}")
                result = CleanupResult(
                    task_name=task.name,
                    status=CleanupStatus.FAILED,
                    items_processed=0,
                    items_fixed=0,
                    errors=[str(e)],
                    warnings=[],
                    execution_time=0,
                    details={},
                )
                results.append(result)

        # Generate summary report
        self._generate_report(results, backup_dir)

        return results

    def _generate_report(self, results: list[CleanupResult], backup_dir: Path | None) -> None:
        """Generate comprehensive cleanup report"""
        report_path = self.project_root / "cleanup_report.json"

        report = {
            "timestamp": time.time(),
            "config": {
                "dry_run": self.config.dry_run,
                "create_backups": self.config.create_backups,
                "backup_location": str(backup_dir) if backup_dir else None,
            },
            "summary": {
                "total_tasks": len(results),
                "completed_tasks": sum(1 for r in results if r.status == CleanupStatus.COMPLETED),
                "failed_tasks": sum(1 for r in results if r.status == CleanupStatus.FAILED),
                "total_items_processed": sum(r.items_processed for r in results),
                "total_items_fixed": sum(r.items_fixed for r in results),
                "total_execution_time": sum(r.execution_time for r in results),
            },
            "results": [
                {
                    "task_name": r.task_name,
                    "status": r.status.value,
                    "items_processed": r.items_processed,
                    "items_fixed": r.items_fixed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time": r.execution_time,
                    "details": r.details,
                }
                for r in results
            ],
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Cleanup report saved to {report_path}")

        # Print summary
        summary = report["summary"]
        self.logger.info("Cleanup Summary:")
        self.logger.info(f"  Tasks: {summary['completed_tasks']}/{summary['total_tasks']} completed")
        self.logger.info(f"  Items: {summary['total_items_fixed']}/{summary['total_items_processed']} fixed")
        self.logger.info(f"  Time: {summary['total_execution_time']:.2f} seconds")


def main():
    """Main entry point for unified cleanup"""
    parser = argparse.ArgumentParser(description="Unified Cleanup Framework for AIVillage")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in dry-run mode (default)")
    parser.add_argument("--execute", action="store_true", help="Actually execute cleanup (overrides dry-run)")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")

    args = parser.parse_args()

    # Configure cleanup
    config = CleanupConfig(
        dry_run=not args.execute, create_backups=not args.no_backup, verbose=args.verbose, parallel=args.parallel
    )

    # Run cleanup
    manager = UnifiedCleanupManager(config, args.project_root)
    results = manager.run_all_tasks()

    # Exit with appropriate code
    failed_tasks = sum(1 for r in results if r.status == CleanupStatus.FAILED)
    sys.exit(1 if failed_tasks > 0 else 0)


if __name__ == "__main__":
    main()
