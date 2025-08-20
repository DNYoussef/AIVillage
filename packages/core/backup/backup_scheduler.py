"""
Automated Backup Scheduler for AIVillage.

Provides automated, scheduled backup operations with intelligent scheduling,
retention management, and health monitoring.
"""

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .backup_manager import BackupManager, BackupType
from .restore_manager import RestoreManager

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduled backups."""

    FULL_DAILY = "full_daily"
    FULL_WEEKLY = "full_weekly"
    INCREMENTAL_HOURLY = "incremental_hourly"
    TENANT_DAILY = "tenant_daily"
    CONFIGURATION_HOURLY = "configuration_hourly"
    EMERGENCY_TRIGGERED = "emergency_triggered"


@dataclass
class ScheduledJob:
    """Scheduled backup job configuration."""

    job_id: str
    schedule_type: ScheduleType
    backup_type: BackupType
    enabled: bool = True

    # Schedule configuration
    hour: int | None = None
    minute: int | None = None
    day_of_week: str | None = None  # monday, tuesday, etc.
    interval_hours: int | None = None

    # Job-specific parameters
    include_tenants: list[str] | None = None
    exclude_components: list[str] | None = None

    # Job state
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Health monitoring
    consecutive_failures: int = 0
    last_error: str | None = None
    average_duration_minutes: float = 0.0


class BackupScheduler:
    """Automated backup scheduler with health monitoring."""

    def __init__(self, backup_manager: BackupManager, restore_manager: RestoreManager):
        """Initialize backup scheduler."""
        self.backup_manager = backup_manager
        self.restore_manager = restore_manager

        self.scheduler_config = self._load_scheduler_config()
        self.scheduled_jobs: dict[str, ScheduledJob] = {}
        self.running = False
        self.scheduler_task: asyncio.Task | None = None

        # Job execution tracking
        self.active_jobs: dict[str, asyncio.Task] = {}
        self.job_history: list[dict[str, Any]] = []

        # Health monitoring
        self.health_checks_enabled = True
        self.alert_callbacks: list[Callable] = []

        # Initialize default schedules
        self._initialize_default_schedules()

        logger.info("Backup scheduler initialized")

    def _load_scheduler_config(self) -> dict[str, Any]:
        """Load scheduler configuration."""
        config_path = Path("config/backup/scheduler_config.json")

        default_config = {
            "default_schedules": {
                "full_backup": {"enabled": True, "schedule": "daily", "hour": 2, "minute": 0, "retention_days": 30},
                "incremental_backup": {"enabled": True, "schedule": "hourly", "interval_hours": 6, "retention_days": 7},
                "tenant_backup": {
                    "enabled": True,
                    "schedule": "daily",
                    "hour": 3,
                    "minute": 30,
                    "retention_per_tenant": 10,
                },
                "config_backup": {"enabled": True, "schedule": "hourly", "interval_hours": 4, "retention_days": 14},
            },
            "health_monitoring": {
                "enabled": True,
                "failure_threshold": 3,
                "alert_after_minutes": 60,
                "health_check_interval": 300,  # 5 minutes
            },
            "retention_policy": {"auto_cleanup": True, "keep_minimum": 5, "cleanup_check_interval_hours": 6},
            "performance": {
                "max_concurrent_jobs": 2,
                "job_timeout_hours": 12,
                "retry_failed_jobs": True,
                "max_retries": 2,
            },
        }

        if config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)

        return default_config

    def _initialize_default_schedules(self):
        """Initialize default scheduled jobs."""
        schedules = self.scheduler_config["default_schedules"]

        # Full daily backup
        if schedules["full_backup"]["enabled"]:
            self.add_scheduled_job(
                job_id="full_daily_backup",
                schedule_type=ScheduleType.FULL_DAILY,
                backup_type=BackupType.FULL,
                hour=schedules["full_backup"]["hour"],
                minute=schedules["full_backup"]["minute"],
            )

        # Incremental backups
        if schedules["incremental_backup"]["enabled"]:
            self.add_scheduled_job(
                job_id="incremental_backup",
                schedule_type=ScheduleType.INCREMENTAL_HOURLY,
                backup_type=BackupType.INCREMENTAL,
                interval_hours=schedules["incremental_backup"]["interval_hours"],
            )

        # Tenant backups
        if schedules["tenant_backup"]["enabled"]:
            self.add_scheduled_job(
                job_id="tenant_daily_backup",
                schedule_type=ScheduleType.TENANT_DAILY,
                backup_type=BackupType.TENANT_ONLY,
                hour=schedules["tenant_backup"]["hour"],
                minute=schedules["tenant_backup"]["minute"],
            )

        # Configuration backups
        if schedules["config_backup"]["enabled"]:
            self.add_scheduled_job(
                job_id="config_backup",
                schedule_type=ScheduleType.CONFIGURATION_HOURLY,
                backup_type=BackupType.CONFIGURATION,
                interval_hours=schedules["config_backup"]["interval_hours"],
            )

    def add_scheduled_job(
        self,
        job_id: str,
        schedule_type: ScheduleType,
        backup_type: BackupType,
        enabled: bool = True,
        hour: int | None = None,
        minute: int | None = None,
        day_of_week: str | None = None,
        interval_hours: int | None = None,
        include_tenants: list[str] | None = None,
        exclude_components: list[str] | None = None,
    ):
        """Add scheduled backup job."""
        job = ScheduledJob(
            job_id=job_id,
            schedule_type=schedule_type,
            backup_type=backup_type,
            enabled=enabled,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week,
            interval_hours=interval_hours,
            include_tenants=include_tenants,
            exclude_components=exclude_components,
        )

        self.scheduled_jobs[job_id] = job
        self._calculate_next_run(job)

        logger.info(f"Added scheduled job: {job_id} ({schedule_type.value})")

    def remove_scheduled_job(self, job_id: str) -> bool:
        """Remove scheduled backup job."""
        if job_id in self.scheduled_jobs:
            # Cancel if running
            if job_id in self.active_jobs:
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]

            del self.scheduled_jobs[job_id]
            logger.info(f"Removed scheduled job: {job_id}")
            return True

        return False

    def enable_job(self, job_id: str) -> bool:
        """Enable scheduled job."""
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id].enabled = True
            self._calculate_next_run(self.scheduled_jobs[job_id])
            logger.info(f"Enabled job: {job_id}")
            return True

        return False

    def disable_job(self, job_id: str) -> bool:
        """Disable scheduled job."""
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id].enabled = False
            self.scheduled_jobs[job_id].next_run = None

            # Cancel if running
            if job_id in self.active_jobs:
                self.active_jobs[job_id].cancel()
                del self.active_jobs[job_id]

            logger.info(f"Disabled job: {job_id}")
            return True

        return False

    def _calculate_next_run(self, job: ScheduledJob):
        """Calculate next run time for job."""
        if not job.enabled:
            job.next_run = None
            return

        now = datetime.utcnow()

        if job.schedule_type in [ScheduleType.FULL_DAILY, ScheduleType.TENANT_DAILY]:
            # Daily at specific time
            next_run = now.replace(hour=job.hour or 0, minute=job.minute or 0, second=0, microsecond=0)

            if next_run <= now:
                next_run += timedelta(days=1)

        elif job.schedule_type == ScheduleType.FULL_WEEKLY:
            # Weekly on specific day - calculate next occurrence
            # Default to Sunday if no day specified
            target_weekday = 6 if not job.day_of_week else {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }.get(job.day_of_week.lower(), 6)

            # Calculate days until next target weekday
            days_ahead = target_weekday - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7

            next_run = now.replace(hour=job.hour or 2, minute=job.minute or 0,
                                 second=0, microsecond=0) + timedelta(days=days_ahead)

        elif job.schedule_type in [ScheduleType.INCREMENTAL_HOURLY, ScheduleType.CONFIGURATION_HOURLY]:
            # Hourly interval
            interval = job.interval_hours or 1

            if job.last_run:
                next_run = job.last_run + timedelta(hours=interval)
            else:
                next_run = now + timedelta(hours=interval)

        else:
            # Default to 1 hour from now
            next_run = now + timedelta(hours=1)

        job.next_run = next_run
        logger.debug(f"Job {job.job_id} next run: {next_run}")

    async def start(self):
        """Start the backup scheduler."""
        if self.running:
            logger.warning("Backup scheduler already running")
            return

        self.running = True
        logger.info("Starting backup scheduler")

        # Start scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Start health monitoring
        if self.health_checks_enabled:
            asyncio.create_task(self._health_monitor_loop())

        # Start retention cleanup
        if self.scheduler_config["retention_policy"]["auto_cleanup"]:
            asyncio.create_task(self._retention_cleanup_loop())

    async def stop(self):
        """Stop the backup scheduler."""
        if not self.running:
            return

        logger.info("Stopping backup scheduler")
        self.running = False

        # Cancel scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all active jobs
        for job_task in list(self.active_jobs.values()):
            job_task.cancel()

        # Wait for jobs to complete
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)

        self.active_jobs.clear()

        logger.info("Backup scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                now = datetime.utcnow()

                # Check each scheduled job
                for job_id, job in self.scheduled_jobs.items():
                    if not job.enabled or not job.next_run:
                        continue

                    # Skip if already running
                    if job_id in self.active_jobs:
                        continue

                    # Check if it's time to run
                    if now >= job.next_run:
                        # Check concurrent job limit
                        max_concurrent = self.scheduler_config["performance"]["max_concurrent_jobs"]
                        if len(self.active_jobs) >= max_concurrent:
                            logger.info(f"Delaying job {job_id} - concurrent limit reached")
                            continue

                        # Start job
                        logger.info(f"Starting scheduled job: {job_id}")
                        task = asyncio.create_task(self._execute_job(job))
                        self.active_jobs[job_id] = task

                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    async def _execute_job(self, job: ScheduledJob):
        """Execute a scheduled backup job."""
        job_start = datetime.utcnow()
        backup_id = None

        try:
            logger.info(f"Executing job {job.job_id} ({job.schedule_type.value})")

            # Execute backup based on type
            if job.backup_type == BackupType.FULL:
                backup_id = await self.backup_manager.create_full_backup(
                    include_tenants=job.include_tenants, exclude_components=job.exclude_components
                )

            elif job.backup_type == BackupType.INCREMENTAL:
                backup_id = await self.backup_manager.create_incremental_backup()

            elif job.backup_type == BackupType.TENANT_ONLY:
                # Create backup for each tenant or all tenants
                if job.include_tenants:
                    for tenant_id in job.include_tenants:
                        tenant_backup_id = await self.backup_manager.create_tenant_backup(tenant_id=tenant_id)
                        if not backup_id:
                            backup_id = tenant_backup_id
                else:
                    # Get all tenants from RBAC system - use empty list if not available
                    logger.warning("Tenant backup without specific tenant list - using empty tenant set")
                    # Create a minimal tenant backup to maintain backup cycle
                    backup_id = await self.backup_manager.create_full_backup(
                        exclude_components=["models", "logs"]
                    )

            elif job.backup_type == BackupType.CONFIGURATION:
                # Configuration-only backup (subset of full backup)
                backup_id = await self.backup_manager.create_full_backup(
                    exclude_components=["models", "logs", "rag_collections"]
                )

            # Update job statistics
            job.last_run = job_start
            job.run_count += 1
            job.success_count += 1
            job.consecutive_failures = 0
            job.last_error = None

            # Calculate average duration
            duration = (datetime.utcnow() - job_start).total_seconds() / 60
            if job.average_duration_minutes == 0:
                job.average_duration_minutes = duration
            else:
                # Weighted average
                job.average_duration_minutes = job.average_duration_minutes * 0.8 + duration * 0.2

            # Calculate next run
            self._calculate_next_run(job)

            # Record success
            self.job_history.append(
                {
                    "job_id": job.job_id,
                    "backup_id": backup_id,
                    "started_at": job_start.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "success": True,
                    "duration_minutes": duration,
                    "error": None,
                }
            )

            logger.info(f"Job {job.job_id} completed successfully: {backup_id}")

        except Exception as e:
            # Update job failure statistics
            job.failure_count += 1
            job.consecutive_failures += 1
            job.last_error = str(e)

            # Record failure
            duration = (datetime.utcnow() - job_start).total_seconds() / 60
            self.job_history.append(
                {
                    "job_id": job.job_id,
                    "backup_id": backup_id,
                    "started_at": job_start.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "success": False,
                    "duration_minutes": duration,
                    "error": str(e),
                }
            )

            logger.error(f"Job {job.job_id} failed: {e}")

            # Trigger alerts if threshold reached
            failure_threshold = self.scheduler_config["health_monitoring"]["failure_threshold"]
            if job.consecutive_failures >= failure_threshold:
                await self._send_alert(
                    f"Backup job {job.job_id} has failed {job.consecutive_failures} consecutive times", "critical"
                )

            # Retry if configured
            if (
                self.scheduler_config["performance"]["retry_failed_jobs"]
                and job.consecutive_failures <= self.scheduler_config["performance"]["max_retries"]
            ):
                # Schedule retry in 30 minutes
                job.next_run = datetime.utcnow() + timedelta(minutes=30)
                logger.info(f"Scheduled retry for job {job.job_id} at {job.next_run}")
            else:
                self._calculate_next_run(job)

        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    async def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self.running:
            try:
                await self._perform_health_checks()

                # Sleep until next health check
                interval = self.scheduler_config["health_monitoring"]["health_check_interval"]
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _perform_health_checks(self):
        """Perform system health checks."""
        now = datetime.utcnow()
        issues = []

        # Check for overdue jobs
        for job_id, job in self.scheduled_jobs.items():
            if not job.enabled or not job.next_run:
                continue

            overdue_minutes = (now - job.next_run).total_seconds() / 60
            if overdue_minutes > 60:  # More than 1 hour overdue
                issues.append(f"Job {job_id} is {overdue_minutes:.1f} minutes overdue")

        # Check for long-running jobs
        self.scheduler_config["performance"]["job_timeout_hours"]
        for job_id, task in self.active_jobs.items():
            # Check if task is still running and not cancelled
            if not task.done() and not task.cancelled():
                # Log long-running jobs for manual review
                logger.info(f"Job {job_id} is still running - monitoring for completion")

        # Check backup success rates
        recent_history = [
            h for h in self.job_history if datetime.fromisoformat(h["started_at"]) > now - timedelta(hours=24)
        ]

        if recent_history:
            success_rate = sum(1 for h in recent_history if h["success"]) / len(recent_history)
            if success_rate < 0.8:  # Less than 80% success rate
                issues.append(f"Low backup success rate: {success_rate:.1%}")

        # Check disk space
        try:
            import shutil
            total_size = sum(f.stat().st_size for f in self.backup_manager.backup_root.rglob("*") if f.is_file())
            free_space = shutil.disk_usage(self.backup_manager.backup_root).free

            # Alert if less than 10GB free space or backup directory is over 80% of available space
            min_free_gb = 10 * 1024 * 1024 * 1024  # 10GB in bytes
            if free_space < min_free_gb:
                issues.append(f"Low disk space: {free_space / (1024**3):.1f}GB free")
            elif total_size > free_space * 0.8:
                issues.append(f"Backup directory consuming {total_size / (1024**3):.1f}GB, approaching capacity")
        except Exception as e:
            issues.append(f"Unable to check disk space: {e}")

        # Send alerts if issues found
        if issues:
            await self._send_alert("Backup system health issues detected:\n" + "\n".join(issues), "warning")

    async def _retention_cleanup_loop(self):
        """Automatic retention cleanup loop."""
        while self.running:
            try:
                await self._cleanup_old_backups()

                # Sleep until next cleanup
                interval_hours = self.scheduler_config["retention_policy"]["cleanup_check_interval_hours"]
                await asyncio.sleep(interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retention cleanup error: {e}")
                await asyncio.sleep(3600)  # 1 hour on error

    async def _cleanup_old_backups(self):
        """Clean up old backups according to retention policy."""
        logger.info("Performing backup retention cleanup")

        # Get all backups
        all_backups = await self.backup_manager.list_backups(limit=1000)
        keep_minimum = self.scheduler_config["retention_policy"]["keep_minimum"]

        # Group by backup type
        backups_by_type = {}
        for backup in all_backups:
            backup_type = backup.backup_type
            if backup_type not in backups_by_type:
                backups_by_type[backup_type] = []
            backups_by_type[backup_type].append(backup)

        # Apply retention policy per type
        for backup_type, backups in backups_by_type.items():
            # Sort by creation time (newest first)
            backups.sort(key=lambda b: b.created_at, reverse=True)

            # Determine retention period
            if backup_type == BackupType.FULL:
                retention_days = self.scheduler_config["default_schedules"]["full_backup"]["retention_days"]
            elif backup_type == BackupType.INCREMENTAL:
                retention_days = self.scheduler_config["default_schedules"]["incremental_backup"]["retention_days"]
            elif backup_type == BackupType.TENANT_ONLY:
                retention_days = 30  # Default for tenant backups
            else:
                retention_days = 14  # Default retention

            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # Find backups to delete
            to_delete = []
            for i, backup in enumerate(backups):
                # Always keep minimum number of backups
                if i < keep_minimum:
                    continue

                # Delete if older than retention period
                if backup.created_at < cutoff_date:
                    to_delete.append(backup)

            # Delete old backups
            for backup in to_delete:
                try:
                    await self._delete_backup(backup.backup_id)
                    logger.info(f"Deleted old backup: {backup.backup_id}")
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup.backup_id}: {e}")

    async def _delete_backup(self, backup_id: str):
        """Delete backup files and metadata."""
        # Find and delete backup files
        for backup_dir in [
            self.backup_manager.full_backups_dir,
            self.backup_manager.incremental_backups_dir,
            self.backup_manager.tenant_backups_dir,
            self.backup_manager.config_backups_dir,
        ]:
            for suffix in [".tar.gz", ".tar.gz.enc"]:
                backup_file = backup_dir / f"{backup_id}{suffix}"
                if backup_file.exists():
                    backup_file.unlink()

        # Remove from database
        conn = sqlite3.connect(self.backup_manager.metadata_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM backups WHERE backup_id = ?", (backup_id,))
        conn.commit()
        conn.close()

    async def _send_alert(self, message: str, severity: str = "info"):
        """Send alert via configured channels."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "severity": severity,
            "component": "backup_scheduler",
        }

        logger.log(
            logging.CRITICAL if severity == "critical" else logging.WARNING if severity == "warning" else logging.INFO,
            f"ALERT: {message}",
        )

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    # Query methods

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of scheduled job."""
        if job_id not in self.scheduled_jobs:
            return None

        job = self.scheduled_jobs[job_id]

        return {
            "job_id": job.job_id,
            "schedule_type": job.schedule_type.value,
            "backup_type": job.backup_type.value,
            "enabled": job.enabled,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "next_run": job.next_run.isoformat() if job.next_run else None,
            "run_count": job.run_count,
            "success_count": job.success_count,
            "failure_count": job.failure_count,
            "consecutive_failures": job.consecutive_failures,
            "average_duration_minutes": job.average_duration_minutes,
            "last_error": job.last_error,
            "currently_running": job_id in self.active_jobs,
        }

    def get_all_jobs_status(self) -> list[dict[str, Any]]:
        """Get status of all scheduled jobs."""
        return [self.get_job_status(job_id) for job_id in self.scheduled_jobs.keys()]

    def get_job_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get job execution history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        return [h for h in self.job_history if datetime.fromisoformat(h["started_at"]) > cutoff]

    def trigger_emergency_backup(self) -> asyncio.Task:
        """Trigger emergency backup immediately."""
        logger.info("Triggering emergency backup")

        emergency_job = ScheduledJob(
            job_id=f"emergency_{int(time.time())}",
            schedule_type=ScheduleType.EMERGENCY_TRIGGERED,
            backup_type=BackupType.EMERGENCY,
        )

        task = asyncio.create_task(self._execute_job(emergency_job))
        return task


async def create_backup_scheduler(backup_manager: BackupManager, restore_manager: RestoreManager) -> BackupScheduler:
    """Create and configure backup scheduler."""
    scheduler = BackupScheduler(backup_manager, restore_manager)

    logger.info("Backup scheduler created")
    return scheduler


if __name__ == "__main__":
    # Example usage
    async def main():
        from .backup_manager import BackupManager
        from .restore_manager import RestoreManager

        # Initialize components
        backup_manager = BackupManager()
        restore_manager = RestoreManager(backup_manager)
        scheduler = await create_backup_scheduler(backup_manager, restore_manager)

        # Add custom alert callback
        async def alert_callback(alert):
            print(f"ALERT: {alert['severity'].upper()} - {alert['message']}")

        scheduler.add_alert_callback(alert_callback)

        # Start scheduler
        await scheduler.start()

        # Let it run for a bit
        await asyncio.sleep(10)

        # Show job status
        jobs = scheduler.get_all_jobs_status()
        for job in jobs:
            print(f"Job: {job['job_id']} - Next run: {job['next_run']}")

        # Stop scheduler
        await scheduler.stop()

    asyncio.run(main())
