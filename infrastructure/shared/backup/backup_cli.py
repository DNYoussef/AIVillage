"""
Backup and Restore Command Line Interface for AIVillage.

Provides comprehensive CLI commands for backup operations, restore procedures,
and system administration.
"""

import asyncio
from datetime import datetime
import json
import sys

try:
    import click
    import tabulate

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    import argparse

from .backup_manager import BackupManager, BackupStatus, BackupType
from .backup_scheduler import create_backup_scheduler
from .restore_manager import RestoreManager, RestoreStatus, RestoreStrategy


class BackupCLI:
    """Backup and restore command line interface."""

    def __init__(self):
        """Initialize CLI with managers."""
        self.backup_manager = None
        self.restore_manager = None
        self.scheduler = None

    async def _initialize_managers(self):
        """Initialize backup system managers."""
        if not self.backup_manager:
            self.backup_manager = BackupManager()
            self.restore_manager = RestoreManager(self.backup_manager)
            self.scheduler = await create_backup_scheduler(self.backup_manager, self.restore_manager)

    async def create_backup(
        self,
        backup_type: str = "full",
        tenants: list[str] | None = None,
        exclude_components: list[str] | None = None,
        output_format: str = "text",
    ):
        """Create a backup."""
        await self._initialize_managers()

        try:
            if backup_type == "full":
                backup_id = await self.backup_manager.create_full_backup(
                    include_tenants=tenants, exclude_components=exclude_components
                )
            elif backup_type == "incremental":
                backup_id = await self.backup_manager.create_incremental_backup()
            elif backup_type == "tenant" and tenants:
                backup_id = await self.backup_manager.create_tenant_backup(tenant_id=tenants[0], include_models=True)
            else:
                self._error("Invalid backup type or missing tenant for tenant backup")
                return

            # Get backup info
            backup_info = await self.backup_manager.get_backup_info(backup_id)

            if output_format == "json":
                self._output_json(
                    {
                        "backup_id": backup_id,
                        "backup_type": backup_info.backup_type.value,
                        "status": backup_info.status.value,
                        "size_mb": backup_info.size_bytes / 1024 / 1024,
                        "file_count": backup_info.file_count,
                        "created_at": backup_info.created_at.isoformat(),
                    }
                )
            else:
                self._success(f"Backup created successfully: {backup_id}")
                self._info(f"Type: {backup_info.backup_type.value}")
                self._info(f"Size: {backup_info.size_bytes / 1024 / 1024:.2f} MB")
                self._info(f"Files: {backup_info.file_count}")
                self._info(f"Status: {backup_info.status.value}")

        except Exception as e:
            self._error(f"Backup failed: {e}")

    async def list_backups(
        self, backup_type: str | None = None, status: str | None = None, limit: int = 20, output_format: str = "table"
    ):
        """List available backups."""
        await self._initialize_managers()

        try:
            backup_type_enum = BackupType(backup_type) if backup_type else None
            status_enum = BackupStatus(status) if status else None

            backups = await self.backup_manager.list_backups(
                backup_type=backup_type_enum, status=status_enum, limit=limit
            )

            if output_format == "json":
                backup_data = []
                for backup in backups:
                    backup_data.append(
                        {
                            "backup_id": backup.backup_id,
                            "type": backup.backup_type.value,
                            "status": backup.status.value,
                            "size_mb": backup.size_bytes / 1024 / 1024,
                            "file_count": backup.file_count,
                            "created_at": backup.created_at.isoformat(),
                            "tenants": backup.tenants_included,
                            "components": backup.components_included,
                        }
                    )
                self._output_json({"backups": backup_data})

            elif output_format == "table":
                if not backups:
                    self._info("No backups found")
                    return

                headers = ["Backup ID", "Type", "Status", "Size (MB)", "Files", "Created", "Tenants", "Components"]

                rows = []
                for backup in backups:
                    rows.append(
                        [
                            backup.backup_id[:20] + "..." if len(backup.backup_id) > 20 else backup.backup_id,
                            backup.backup_type.value,
                            backup.status.value,
                            f"{backup.size_bytes / 1024 / 1024:.1f}",
                            str(backup.file_count),
                            backup.created_at.strftime("%Y-%m-%d %H:%M"),
                            ",".join(backup.tenants_included[:2]) + ("..." if len(backup.tenants_included) > 2 else ""),
                            str(len(backup.components_included)),
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

        except Exception as e:
            self._error(f"Failed to list backups: {e}")

    async def backup_info(self, backup_id: str, output_format: str = "text"):
        """Show detailed backup information."""
        await self._initialize_managers()

        try:
            backup_info = await self.backup_manager.get_backup_info(backup_id)

            if not backup_info:
                self._error(f"Backup {backup_id} not found")
                return

            if output_format == "json":
                data = {
                    "backup_id": backup_info.backup_id,
                    "backup_type": backup_info.backup_type.value,
                    "status": backup_info.status.value,
                    "created_at": backup_info.created_at.isoformat(),
                    "completed_at": backup_info.completed_at.isoformat() if backup_info.completed_at else None,
                    "size_bytes": backup_info.size_bytes,
                    "file_count": backup_info.file_count,
                    "checksum": backup_info.checksum,
                    "encrypted": backup_info.encrypted,
                    "compression_ratio": backup_info.compression_ratio,
                    "tenants_included": backup_info.tenants_included,
                    "components_included": backup_info.components_included,
                    "errors": backup_info.errors,
                    "warnings": backup_info.warnings,
                }
                self._output_json(data)
            else:
                self._info(f"Backup ID: {backup_info.backup_id}")
                self._info(f"Type: {backup_info.backup_type.value}")
                self._info(f"Status: {backup_info.status.value}")
                self._info(f"Created: {backup_info.created_at}")
                if backup_info.completed_at:
                    self._info(f"Completed: {backup_info.completed_at}")
                    duration = backup_info.completed_at - backup_info.created_at
                    self._info(f"Duration: {duration}")

                self._info(f"Size: {backup_info.size_bytes / 1024 / 1024:.2f} MB")
                self._info(f"Files: {backup_info.file_count}")
                self._info(f"Encrypted: {backup_info.encrypted}")
                self._info(f"Checksum: {backup_info.checksum}")

                if backup_info.tenants_included:
                    self._info(f"Tenants: {', '.join(backup_info.tenants_included)}")

                self._info(f"Components: {', '.join(backup_info.components_included)}")

                if backup_info.errors:
                    self._warning("Errors:")
                    for error in backup_info.errors:
                        self._warning(f"  - {error}")

                if backup_info.warnings:
                    self._warning("Warnings:")
                    for warning in backup_info.warnings:
                        self._warning(f"  - {warning}")

        except Exception as e:
            self._error(f"Failed to get backup info: {e}")

    async def restore_backup(
        self,
        backup_id: str,
        restore_type: str = "full",
        strategy: str = "replace",
        tenant_id: str | None = None,
        component: str | None = None,
        create_rollback: bool = True,
        dry_run: bool = False,
    ):
        """Restore from backup."""
        await self._initialize_managers()

        try:
            if dry_run:
                self._info("DRY RUN - No actual restore will be performed")
                backup_info = await self.backup_manager.get_backup_info(backup_id)
                if not backup_info:
                    self._error(f"Backup {backup_id} not found")
                    return

                self._info(f"Would restore backup: {backup_id}")
                self._info(f"Restore type: {restore_type}")
                self._info(f"Strategy: {strategy}")
                self._info(f"Components: {', '.join(backup_info.components_included)}")
                return

            restore_strategy = RestoreStrategy(strategy)

            if restore_type == "full":
                restore_id = await self.restore_manager.restore_full_system(
                    backup_id=backup_id, strategy=restore_strategy, create_rollback=create_rollback
                )
            elif restore_type == "tenant":
                if not tenant_id:
                    self._error("Tenant ID required for tenant restore")
                    return

                restore_id = await self.restore_manager.restore_tenant(
                    backup_id=backup_id, tenant_id=tenant_id, strategy=restore_strategy
                )
            elif restore_type == "component":
                if not component:
                    self._error("Component name required for component restore")
                    return

                restore_id = await self.restore_manager.restore_component(
                    backup_id=backup_id, component=component, strategy=restore_strategy
                )
            else:
                self._error(f"Invalid restore type: {restore_type}")
                return

            self._success(f"Restore started: {restore_id}")

            # Monitor restore progress
            while True:
                status = await self.restore_manager.get_restore_status(restore_id)
                if not status:
                    break

                if status.status in [RestoreStatus.COMPLETED, RestoreStatus.FAILED, RestoreStatus.ROLLED_BACK]:
                    break

                self._info(f"Restore status: {status.status.value} ({status.progress_percent:.1f}%)")
                await asyncio.sleep(5)

            # Final status
            final_status = await self.restore_manager.get_restore_status(restore_id)
            if final_status:
                if final_status.status == RestoreStatus.COMPLETED:
                    self._success("Restore completed successfully")
                    if final_status.verification_results:
                        for component, verified in final_status.verification_results.items():
                            status_text = "✓" if verified else "✗"
                            self._info(f"  {status_text} {component}")
                elif final_status.status == RestoreStatus.FAILED:
                    self._error("Restore failed")
                    for error in final_status.errors:
                        self._error(f"  - {error}")
                elif final_status.status == RestoreStatus.ROLLED_BACK:
                    self._warning("Restore was rolled back due to failure")

        except Exception as e:
            self._error(f"Restore failed: {e}")

    async def list_restores(self, limit: int = 10, output_format: str = "table"):
        """List restore operations."""
        await self._initialize_managers()

        try:
            restores = await self.restore_manager.list_restores(limit=limit)

            if output_format == "json":
                restore_data = []
                for restore in restores:
                    restore_data.append(
                        {
                            "restore_id": restore.restore_id,
                            "backup_id": restore.backup_id,
                            "type": restore.restore_type.value,
                            "strategy": restore.strategy.value,
                            "status": restore.status.value,
                            "progress": restore.progress_percent,
                            "created_at": restore.created_at.isoformat(),
                            "completed_at": restore.completed_at.isoformat() if restore.completed_at else None,
                        }
                    )
                self._output_json({"restores": restore_data})

            elif output_format == "table":
                if not restores:
                    self._info("No restore operations found")
                    return

                headers = ["Restore ID", "Backup ID", "Type", "Strategy", "Status", "Progress", "Created"]

                rows = []
                for restore in restores:
                    rows.append(
                        [
                            restore.restore_id[:20] + "..." if len(restore.restore_id) > 20 else restore.restore_id,
                            restore.backup_id[:15] + "..." if len(restore.backup_id) > 15 else restore.backup_id,
                            restore.restore_type.value,
                            restore.strategy.value,
                            restore.status.value,
                            f"{restore.progress_percent:.1f}%",
                            restore.created_at.strftime("%Y-%m-%d %H:%M"),
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

        except Exception as e:
            self._error(f"Failed to list restores: {e}")

    async def scheduler_status(self, output_format: str = "table"):
        """Show backup scheduler status."""
        await self._initialize_managers()

        try:
            jobs = self.scheduler.get_all_jobs_status()

            if output_format == "json":
                self._output_json({"jobs": jobs})

            elif output_format == "table":
                if not jobs:
                    self._info("No scheduled jobs")
                    return

                headers = ["Job ID", "Type", "Enabled", "Next Run", "Success Rate", "Avg Duration", "Status"]

                rows = []
                for job in jobs:
                    success_rate = job["success_count"] / max(job["run_count"], 1) * 100 if job["run_count"] > 0 else 0

                    next_run = (
                        datetime.fromisoformat(job["next_run"]).strftime("%m-%d %H:%M") if job["next_run"] else "N/A"
                    )

                    status = "Running" if job["currently_running"] else "Scheduled"
                    if not job["enabled"]:
                        status = "Disabled"

                    rows.append(
                        [
                            job["job_id"],
                            job["schedule_type"],
                            "Yes" if job["enabled"] else "No",
                            next_run,
                            f"{success_rate:.1f}%",
                            f"{job['average_duration_minutes']:.1f}m",
                            status,
                        ]
                    )

                table = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
                print(table)

                # Show recent history
                history = self.scheduler.get_job_history(hours=24)
                if history:
                    print("\nRecent Activity (last 24 hours):")
                    for entry in history[-5:]:  # Last 5 entries
                        status_icon = "✓" if entry["success"] else "✗"
                        timestamp = datetime.fromisoformat(entry["started_at"]).strftime("%H:%M")
                        duration = f"{entry['duration_minutes']:.1f}m"
                        print(f"  {status_icon} {timestamp} - {entry['job_id']} ({duration})")

        except Exception as e:
            self._error(f"Failed to get scheduler status: {e}")

    async def start_scheduler(self):
        """Start the backup scheduler."""
        await self._initialize_managers()

        try:
            await self.scheduler.start()
            self._success("Backup scheduler started")

            # Show current jobs
            jobs = self.scheduler.get_all_jobs_status()
            enabled_jobs = [j for j in jobs if j["enabled"]]

            if enabled_jobs:
                self._info(f"Active scheduled jobs: {len(enabled_jobs)}")
                for job in enabled_jobs:
                    next_run = (
                        datetime.fromisoformat(job["next_run"]).strftime("%Y-%m-%d %H:%M") if job["next_run"] else "N/A"
                    )
                    self._info(f"  - {job['job_id']}: next run at {next_run}")
            else:
                self._warning("No enabled scheduled jobs")

        except Exception as e:
            self._error(f"Failed to start scheduler: {e}")

    async def stop_scheduler(self):
        """Stop the backup scheduler."""
        await self._initialize_managers()

        try:
            await self.scheduler.stop()
            self._success("Backup scheduler stopped")
        except Exception as e:
            self._error(f"Failed to stop scheduler: {e}")

    async def emergency_backup(self):
        """Trigger emergency backup."""
        await self._initialize_managers()

        try:
            self._info("Triggering emergency backup...")
            task = self.scheduler.trigger_emergency_backup()

            # Wait for completion (with timeout)
            try:
                await asyncio.wait_for(task, timeout=3600)  # 1 hour timeout
                self._success("Emergency backup completed")
            except asyncio.TimeoutError:
                self._warning("Emergency backup is taking longer than expected")
                self._info("Check backup status with: backup list")

        except Exception as e:
            self._error(f"Emergency backup failed: {e}")

    def _success(self, message: str):
        """Print success message."""
        print(f"✓ {message}")

    def _info(self, message: str):
        """Print info message."""
        print(f"ℹ {message}")

    def _warning(self, message: str):
        """Print warning message."""
        print(f"⚠ {message}")

    def _error(self, message: str):
        """Print error message."""
        print(f"✗ {message}", file=sys.stderr)

    def _output_json(self, data):
        """Output data as JSON."""
        print(json.dumps(data, indent=2, default=str))


# CLI command definitions
if CLICK_AVAILABLE:
    # Click-based CLI
    @click.group()
    def backup_cli():
        """AIVillage Backup and Restore Management"""
        pass

    @backup_cli.command()
    @click.option(
        "--type",
        "backup_type",
        default="full",
        type=click.Choice(["full", "incremental", "tenant"]),
        help="Type of backup to create",
    )
    @click.option("--tenant", multiple=True, help="Tenant IDs to include (for tenant backup)")
    @click.option("--exclude", multiple=True, help="Components to exclude")
    @click.option(
        "--format", "output_format", default="text", type=click.Choice(["text", "json"]), help="Output format"
    )
    def create(backup_type, tenant, exclude, output_format):
        """Create a backup"""
        cli = BackupCLI()
        asyncio.run(
            cli.create_backup(
                backup_type=backup_type,
                tenants=list(tenant) if tenant else None,
                exclude_components=list(exclude) if exclude else None,
                output_format=output_format,
            )
        )

    @backup_cli.command()
    @click.option("--type", "backup_type", help="Filter by backup type")
    @click.option("--status", help="Filter by status")
    @click.option("--limit", default=20, help="Maximum number of backups to show")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def list(backup_type, status, limit, output_format):
        """List available backups"""
        cli = BackupCLI()
        asyncio.run(cli.list_backups(backup_type=backup_type, status=status, limit=limit, output_format=output_format))

    @backup_cli.command()
    @click.argument("backup_id")
    @click.option(
        "--format", "output_format", default="text", type=click.Choice(["text", "json"]), help="Output format"
    )
    def info(backup_id, output_format):
        """Show detailed backup information"""
        cli = BackupCLI()
        asyncio.run(cli.backup_info(backup_id, output_format))

    @backup_cli.command()
    @click.argument("backup_id")
    @click.option(
        "--type",
        "restore_type",
        default="full",
        type=click.Choice(["full", "tenant", "component"]),
        help="Type of restore",
    )
    @click.option(
        "--strategy",
        default="replace",
        type=click.Choice(["replace", "merge", "side_by_side", "test_restore"]),
        help="Restore strategy",
    )
    @click.option("--tenant", help="Tenant ID (for tenant restore)")
    @click.option("--component", help="Component name (for component restore)")
    @click.option("--no-rollback", is_flag=True, help="Skip creating rollback point")
    @click.option("--dry-run", is_flag=True, help="Show what would be restored without doing it")
    def restore(backup_id, restore_type, strategy, tenant, component, no_rollback, dry_run):
        """Restore from backup"""
        cli = BackupCLI()
        asyncio.run(
            cli.restore_backup(
                backup_id=backup_id,
                restore_type=restore_type,
                strategy=strategy,
                tenant_id=tenant,
                component=component,
                create_rollback=not no_rollback,
                dry_run=dry_run,
            )
        )

    @backup_cli.command("list-restores")
    @click.option("--limit", default=10, help="Maximum number of restores to show")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def list_restores(limit, output_format):
        """List restore operations"""
        cli = BackupCLI()
        asyncio.run(cli.list_restores(limit, output_format))

    @backup_cli.command("scheduler")
    @click.option(
        "--format", "output_format", default="table", type=click.Choice(["table", "json"]), help="Output format"
    )
    def scheduler_status(output_format):
        """Show backup scheduler status"""
        cli = BackupCLI()
        asyncio.run(cli.scheduler_status(output_format))

    @backup_cli.command("start-scheduler")
    def start_scheduler():
        """Start the backup scheduler"""
        cli = BackupCLI()
        asyncio.run(cli.start_scheduler())

    @backup_cli.command("stop-scheduler")
    def stop_scheduler():
        """Stop the backup scheduler"""
        cli = BackupCLI()
        asyncio.run(cli.stop_scheduler())

    @backup_cli.command("emergency")
    def emergency():
        """Trigger emergency backup"""
        cli = BackupCLI()
        asyncio.run(cli.emergency_backup())

    if __name__ == "__main__":
        backup_cli()

else:
    # Fallback argparse-based CLI
    def main():
        parser = argparse.ArgumentParser(description="AIVillage Backup and Restore Management")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Create backup
        create_parser = subparsers.add_parser("create", help="Create a backup")
        create_parser.add_argument(
            "--type", choices=["full", "incremental", "tenant"], default="full", help="Type of backup"
        )
        create_parser.add_argument("--tenant", action="append", help="Tenant IDs")
        create_parser.add_argument("--exclude", action="append", help="Components to exclude")

        # List backups
        list_parser = subparsers.add_parser("list", help="List backups")
        list_parser.add_argument("--limit", type=int, default=20, help="Maximum number to show")

        # Emergency backup
        subparsers.add_parser("emergency", help="Emergency backup")

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        cli = BackupCLI()

        if args.command == "create":
            asyncio.run(cli.create_backup(backup_type=args.type, tenants=args.tenant, exclude_components=args.exclude))
        elif args.command == "list":
            asyncio.run(cli.list_backups(limit=args.limit))
        elif args.command == "emergency":
            asyncio.run(cli.emergency_backup())
        else:
            print(f"Unknown command: {args.command}")

    if __name__ == "__main__":
        main()
