#!/usr/bin/env python3
"""
Complete Phase 2 Migration Script

This script completes the remaining fog infrastructure tasks and Phase 2.3: monitoring migration.

Tasks:
1. Complete remaining fog components migration
2. Move libs/edge/* → infrastructure/fog/edge/ (merge with existing)
3. Move libs/monitoring/* → devops/monitoring/ (merge with existing)
4. Handle monitoring-related files in libs/core/

Author: Claude Code Agent
"""

import json
import shutil
from pathlib import Path


class Phase2MigrationManager:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.migration_log = []
        self.errors = []

    def log_action(self, action: str, source: str = "", target: str = "", status: str = "success"):
        """Log migration action"""
        entry = {"action": action, "source": source, "target": target, "status": status, "timestamp": str(Path.cwd())}
        self.migration_log.append(entry)
        print(f"[{status.upper()}] {action}: {source} -> {target}")

    def safe_move(self, source: Path, target: Path, merge_mode: bool = False) -> bool:
        """Safely move files/directories with merge support"""
        try:
            if not source.exists():
                self.log_action("SKIP", str(source), str(target), "not_found")
                return False

            # Create parent directories
            target.parent.mkdir(parents=True, exist_ok=True)

            if merge_mode and target.exists():
                if source.is_dir() and target.is_dir():
                    # Merge directories
                    return self.merge_directories(source, target)
                else:
                    self.log_action("SKIP", str(source), str(target), "already_exists")
                    return False
            else:
                # Direct move
                if target.exists():
                    self.log_action("SKIP", str(source), str(target), "already_exists")
                    return False

                shutil.move(str(source), str(target))
                self.log_action("MOVE", str(source), str(target))
                return True

        except Exception as e:
            self.errors.append(f"Error moving {source} to {target}: {str(e)}")
            self.log_action("ERROR", str(source), str(target), "failed")
            return False

    def merge_directories(self, source: Path, target: Path) -> bool:
        """Merge source directory into target directory"""
        try:
            merged_files = 0
            skipped_files = 0

            for item in source.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(source)
                    target_file = target / relative_path

                    if not target_file.exists():
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(item), str(target_file))
                        merged_files += 1
                    else:
                        skipped_files += 1

            # Remove source directory if empty after merge
            try:
                shutil.rmtree(str(source))
                self.log_action("MERGE", str(source), str(target), f"merged_{merged_files}_skipped_{skipped_files}")
            except OSError:
                self.log_action(
                    "MERGE_PARTIAL", str(source), str(target), f"merged_{merged_files}_skipped_{skipped_files}"
                )

            return True

        except Exception as e:
            self.errors.append(f"Error merging {source} into {target}: {str(e)}")
            return False

    def complete_fog_infrastructure(self):
        """Complete remaining fog infrastructure migrations"""
        print("\n=== Completing Fog Infrastructure Migration ===")

        # 1. Check and complete any remaining fog components
        libs_fog = self.root / "libs" / "fog"
        infra_fog = self.root / "infrastructure" / "fog"

        if libs_fog.exists():
            print(f"Found remaining fog components in: {libs_fog}")

            # Move bridges if they don't exist in infrastructure
            fog_bridges_src = libs_fog / "bridges"
            fog_bridges_dst = infra_fog / "bridges"
            if fog_bridges_src.exists():
                self.safe_move(fog_bridges_src, fog_bridges_dst, merge_mode=True)

            # Move examples to appropriate location
            fog_examples_src = libs_fog / "examples"
            fog_examples_dst = self.root / "examples" / "fog"
            if fog_examples_src.exists():
                self.safe_move(fog_examples_src, fog_examples_dst, merge_mode=True)

            # Move gateway components (should already be moved, but check)
            fog_gateway_src = libs_fog / "gateway"
            self.root / "infrastructure" / "gateway"
            if fog_gateway_src.exists():
                print("Warning: Found gateway components in libs/fog - these should have been moved already")
                # Don't move gateway as it's already handled

        # 2. Move libs/edge/* → infrastructure/fog/edge/ (merge with existing)
        libs_edge = self.root / "libs" / "edge"
        infra_fog_edge = infra_fog / "edge"

        if libs_edge.exists():
            print(f"Migrating edge components: {libs_edge} -> {infra_fog_edge}")
            self.safe_move(libs_edge, infra_fog_edge, merge_mode=True)
        else:
            print("No edge components found in libs/edge")

    def migrate_monitoring(self):
        """Phase 2.3: Move monitoring to devops"""
        print("\n=== Phase 2.3: Monitoring Migration ===")

        # 1. Move libs/monitoring/* → devops/monitoring/ (merge with existing)
        libs_monitoring = self.root / "libs" / "monitoring"
        devops_monitoring = self.root / "devops" / "monitoring"

        if libs_monitoring.exists():
            print(f"Migrating monitoring components: {libs_monitoring} -> {devops_monitoring}")
            self.safe_move(libs_monitoring, devops_monitoring, merge_mode=True)
        else:
            print("No monitoring components found in libs/monitoring")

        # 2. Handle monitoring-related files in libs/core/
        libs_core = self.root / "libs" / "core"
        monitoring_related_files = [
            "cost_management/cost_governance_dashboard.py",
            "cost_management/distributed_cost_tracker.py",
            "operations/artifact_collector.py",
            "backup/backup_scheduler.py",
            "legacy/resources/resource_monitor.py",
            "legacy/testing/coverage_harness.py",
        ]

        for rel_path in monitoring_related_files:
            source_file = libs_core / rel_path
            if source_file.exists():
                # Determine appropriate target in devops
                if "cost_management" in rel_path:
                    target_file = devops_monitoring / "cost" / source_file.name
                elif "operations" in rel_path:
                    target_file = devops_monitoring / "operations" / source_file.name
                elif "backup" in rel_path:
                    target_file = devops_monitoring / "backup" / source_file.name
                elif "legacy/resources" in rel_path:
                    target_file = devops_monitoring / "resources" / source_file.name
                elif "legacy/testing" in rel_path:
                    target_file = devops_monitoring / "testing" / source_file.name
                else:
                    target_file = devops_monitoring / "misc" / source_file.name

                self.safe_move(source_file, target_file)

    def ensure_directory_structure(self):
        """Ensure proper directory structures for migrated components"""
        print("\n=== Ensuring Directory Structures ===")

        directories_to_create = [
            "devops/monitoring/cost",
            "devops/monitoring/operations",
            "devops/monitoring/backup",
            "devops/monitoring/resources",
            "devops/monitoring/testing",
            "devops/monitoring/misc",
            "devops/monitoring/grafana",
            "devops/monitoring/prometheus",
            "devops/monitoring/alerts",
            "infrastructure/fog/bridges",
            "infrastructure/fog/edge",
            "infrastructure/fog/scheduler",
            "infrastructure/fog/monitoring",
            "examples/fog",
        ]

        for dir_path in directories_to_create:
            full_path = self.root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.log_action("MKDIR", "", str(full_path))

    def generate_migration_report(self):
        """Generate migration report"""
        report_path = self.root / "scripts" / "phase2_migration_report.json"

        report = {
            "migration_type": "Phase 2 Completion: Fog Infrastructure + Monitoring Migration",
            "timestamp": str(Path.cwd()),
            "total_actions": len(self.migration_log),
            "errors": len(self.errors),
            "actions": self.migration_log,
            "errors": self.errors,
            "summary": {
                "moves": len([a for a in self.migration_log if a["action"] == "MOVE"]),
                "merges": len([a for a in self.migration_log if a["action"] == "MERGE"]),
                "skips": len([a for a in self.migration_log if a["status"] == "already_exists"]),
                "directories_created": len([a for a in self.migration_log if a["action"] == "MKDIR"]),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n=== Migration Report Generated ===")
        print(f"Report saved to: {report_path}")
        print(f"Total actions: {report['total_actions']}")
        print(f"Successful moves: {report['summary']['moves']}")
        print(f"Successful merges: {report['summary']['merges']}")
        print(f"Skipped (already exist): {report['summary']['skips']}")
        print(f"Directories created: {report['summary']['directories_created']}")
        print(f"Errors: {len(self.errors)}")

        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error}")

    def run_migration(self):
        """Run the complete Phase 2 migration"""
        print("Starting Phase 2 Migration Completion...")
        print(f"Working directory: {self.root}")

        try:
            # Ensure directory structures first
            self.ensure_directory_structure()

            # Complete fog infrastructure
            self.complete_fog_infrastructure()

            # Migrate monitoring
            self.migrate_monitoring()

            # Generate report
            self.generate_migration_report()

            print("\n=== Phase 2 Migration Completion Finished ===")
            return True

        except Exception as e:
            print(f"Critical error during migration: {e}")
            self.errors.append(f"Critical error: {str(e)}")
            return False


def main():
    root_path = Path(__file__).parent.parent
    migrator = Phase2MigrationManager(root_path)
    success = migrator.run_migration()

    if success and not migrator.errors:
        print("✅ Phase 2 migration completed successfully!")
    elif success and migrator.errors:
        print("⚠️  Phase 2 migration completed with some errors.")
    else:
        print("❌ Phase 2 migration failed.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
