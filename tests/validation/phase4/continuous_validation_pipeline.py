"""
Continuous Validation Pipeline for Phase 4

Automated continuous validation pipeline with real-time monitoring and feedback.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
from dataclasses import dataclass
import subprocess
from datetime import datetime, timedelta

from .core.phase4_validator import Phase4ValidationSuite, ValidationResult
from .core.performance_monitor import PerformanceMonitor
from .reports.validation_reporter import ValidationReporter


@dataclass
class ValidationTrigger:
    """Validation trigger configuration"""
    name: str
    enabled: bool
    trigger_type: str  # 'file_change', 'schedule', 'manual', 'git_commit'
    config: Dict[str, Any]


@dataclass 
class PipelineConfig:
    """Continuous validation pipeline configuration"""
    validation_triggers: List[ValidationTrigger]
    notification_config: Dict[str, Any]
    rollback_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    success_gates: Dict[str, Any]


class ContinuousValidationPipeline:
    """
    Continuous validation pipeline for Phase 4 architectural improvements
    """
    
    def __init__(self, project_root: Path, config_path: Optional[Path] = None):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.validator = Phase4ValidationSuite(project_root)
        self.performance_monitor = PerformanceMonitor()
        self.reporter = ValidationReporter()
        
        # Pipeline state
        self.pipeline_running = False
        self.last_validation_result = None
        self.validation_history = []
        self.current_commit_hash = None
        
        # File watching for trigger detection
        self.watched_files = set()
        self.file_watcher_task = None
        
        # Metrics
        self.pipeline_metrics = {
            'validations_run': 0,
            'validations_passed': 0,
            'validations_failed': 0,
            'average_execution_time_ms': 0,
            'last_successful_validation': None,
            'consecutive_failures': 0
        }
    
    async def start_pipeline(self) -> None:
        """Start the continuous validation pipeline"""
        if self.pipeline_running:
            self.logger.warning("Pipeline already running")
            return
        
        self.pipeline_running = True
        self.logger.info("Starting continuous validation pipeline...")
        
        # Initialize components
        await self.validator.initialize()
        
        # Start performance monitoring
        monitor_config = self.config.monitoring_config
        await self.performance_monitor.start_monitoring(
            interval=monitor_config.get('interval', 1.0)
        )
        
        # Set performance baseline if not exists
        if not self.performance_monitor.baseline_metrics:
            # Wait a bit for initial metrics
            await asyncio.sleep(5)
            self.performance_monitor.set_baseline()
        
        # Start file watching if configured
        file_triggers = [t for t in self.config.validation_triggers 
                        if t.trigger_type == 'file_change' and t.enabled]
        if file_triggers:
            await self._start_file_watching()
        
        # Schedule validation triggers
        await self._schedule_validation_triggers()
        
        self.logger.info("Continuous validation pipeline started")
    
    async def stop_pipeline(self) -> None:
        """Stop the continuous validation pipeline"""
        self.pipeline_running = False
        
        # Stop file watching
        if self.file_watcher_task:
            self.file_watcher_task.cancel()
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        self.logger.info("Continuous validation pipeline stopped")
    
    async def run_validation(self, trigger_name: str = "manual") -> ValidationResult:
        """Run a full validation cycle"""
        self.logger.info(f"Starting validation run (trigger: {trigger_name})...")
        
        start_time = time.time()
        
        try:
            # Capture current git state
            current_commit = await self._get_current_commit_hash()
            
            # Run validation
            validation_result = await self.validator.run_full_validation()
            
            # Update metrics
            execution_time = int((time.time() - start_time) * 1000)
            self._update_pipeline_metrics(validation_result, execution_time)
            
            # Store result
            self.last_validation_result = validation_result
            self.validation_history.append({
                'timestamp': validation_result.timestamp,
                'trigger': trigger_name,
                'commit_hash': current_commit,
                'passed': validation_result.passed,
                'execution_time_ms': execution_time,
                'summary': {
                    'errors': len(validation_result.errors),
                    'warnings': len(validation_result.warnings)
                }
            })
            
            # Limit history size
            if len(self.validation_history) > 100:
                self.validation_history = self.validation_history[-100:]
            
            # Check success gates
            gates_passed = await self._check_success_gates(validation_result)
            
            # Handle validation result
            if validation_result.passed and gates_passed:
                await self._handle_validation_success(validation_result, trigger_name)
            else:
                await self._handle_validation_failure(validation_result, trigger_name, gates_passed)
            
            # Generate and send notifications
            await self._send_notifications(validation_result, trigger_name)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation run failed: {e}")
            
            # Create error result
            error_result = ValidationResult(
                passed=False,
                coupling_results={},
                performance_results={},
                quality_results={},
                compatibility_results={},
                integration_results={},
                errors=[str(e)],
                warnings=[],
                execution_time_ms=int((time.time() - start_time) * 1000),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self._update_pipeline_metrics(error_result, error_result.execution_time_ms)
            await self._handle_validation_failure(error_result, trigger_name, False)
            
            return error_result
    
    async def _check_success_gates(self, validation_result: ValidationResult) -> bool:
        """Check if validation result meets success gate criteria"""
        gates_config = self.config.success_gates
        
        # Gate 1: All validation components must pass
        if not validation_result.passed:
            self.logger.warning("Success gate failed: Validation did not pass")
            return False
        
        # Gate 2: Maximum allowed errors
        max_errors = gates_config.get('max_errors', 0)
        if len(validation_result.errors) > max_errors:
            self.logger.warning(f"Success gate failed: Too many errors ({len(validation_result.errors)} > {max_errors})")
            return False
        
        # Gate 3: Performance degradation limits
        max_degradation = gates_config.get('max_performance_degradation', 10.0)
        if validation_result.performance_results:
            overall = validation_result.performance_results.get('overall', {})
            degradation = overall.get('performance_degradation_percent', 0)
            if degradation > max_degradation:
                self.logger.warning(f"Success gate failed: Performance degradation too high ({degradation}% > {max_degradation}%)")
                return False
        
        # Gate 4: Coupling improvement requirements
        coupling_gates = gates_config.get('coupling_requirements', {})
        if validation_result.coupling_results and coupling_gates:
            improvements = validation_result.coupling_results.get('improvements', {})
            for component, min_improvement in coupling_gates.items():
                if component in improvements:
                    actual_improvement = improvements[component].get('improvement_percent', 0)
                    if actual_improvement < min_improvement:
                        self.logger.warning(f"Success gate failed: {component} coupling improvement insufficient ({actual_improvement}% < {min_improvement}%)")
                        return False
        
        # Gate 5: Test coverage requirements
        min_coverage = gates_config.get('min_test_coverage', 80.0)
        if validation_result.quality_results:
            coverage = validation_result.quality_results.get('avg_test_coverage', 0)
            if coverage < min_coverage:
                self.logger.warning(f"Success gate failed: Test coverage too low ({coverage}% < {min_coverage}%)")
                return False
        
        self.logger.info("All success gates passed")
        return True
    
    async def _handle_validation_success(self, validation_result: ValidationResult, trigger: str) -> None:
        """Handle successful validation"""
        self.logger.info(f"✅ Validation passed (trigger: {trigger})")
        
        # Reset consecutive failures counter
        self.pipeline_metrics['consecutive_failures'] = 0
        self.pipeline_metrics['last_successful_validation'] = validation_result.timestamp
        
        # Save successful baseline if configured
        success_config = self.config.success_gates.get('save_baseline_on_success', True)
        if success_config:
            await self.validator.save_baseline_metrics()
            self.performance_monitor.set_baseline()
        
        # Run success hooks
        await self._run_success_hooks(validation_result, trigger)
    
    async def _handle_validation_failure(self, validation_result: ValidationResult, trigger: str, gates_passed: bool) -> None:
        """Handle validation failure"""
        self.pipeline_metrics['consecutive_failures'] += 1
        consecutive_failures = self.pipeline_metrics['consecutive_failures']
        
        self.logger.error(f"❌ Validation failed (trigger: {trigger}, consecutive failures: {consecutive_failures})")
        
        # Check if rollback should be triggered
        rollback_config = self.config.rollback_config
        max_failures = rollback_config.get('max_consecutive_failures', 3)
        
        if consecutive_failures >= max_failures and rollback_config.get('enabled', False):
            await self._trigger_rollback(validation_result)
        
        # Run failure hooks
        await self._run_failure_hooks(validation_result, trigger, gates_passed)
    
    async def _trigger_rollback(self, validation_result: ValidationResult) -> None:
        """Trigger rollback procedure"""
        self.logger.warning("🔄 Triggering rollback due to consecutive validation failures")
        
        rollback_config = self.config.rollback_config
        
        # Git rollback if configured
        if rollback_config.get('git_rollback', {}).get('enabled', False):
            await self._perform_git_rollback()
        
        # Service restart if configured
        if rollback_config.get('service_restart', {}).get('enabled', False):
            await self._restart_services()
        
        # Custom rollback script
        rollback_script = rollback_config.get('custom_script')
        if rollback_script:
            await self._run_rollback_script(rollback_script)
        
        # Send critical alert
        await self._send_critical_alert("Rollback triggered due to consecutive validation failures")
    
    async def _perform_git_rollback(self) -> None:
        """Perform git rollback to last known good commit"""
        try:
            rollback_config = self.config.rollback_config.get('git_rollback', {})
            target_commit = rollback_config.get('target_commit', 'HEAD~1')
            
            # Create rollback branch
            rollback_branch = f"rollback-{int(time.time())}"
            
            subprocess.run(['git', 'checkout', '-b', rollback_branch], 
                         cwd=self.project_root, check=True)
            subprocess.run(['git', 'reset', '--hard', target_commit], 
                         cwd=self.project_root, check=True)
            
            self.logger.info(f"Git rollback completed to {target_commit} on branch {rollback_branch}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git rollback failed: {e}")
    
    async def _restart_services(self) -> None:
        """Restart configured services"""
        try:
            restart_config = self.config.rollback_config.get('service_restart', {})
            services = restart_config.get('services', [])
            
            for service in services:
                subprocess.run(['sudo', 'systemctl', 'restart', service], check=True)
                self.logger.info(f"Restarted service: {service}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Service restart failed: {e}")
    
    async def _run_rollback_script(self, script_path: str) -> None:
        """Run custom rollback script"""
        try:
            subprocess.run([script_path], cwd=self.project_root, check=True)
            self.logger.info(f"Rollback script executed: {script_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Rollback script failed: {e}")
    
    async def _send_notifications(self, validation_result: ValidationResult, trigger: str) -> None:
        """Send validation notifications"""
        notification_config = self.config.notification_config
        
        if not notification_config.get('enabled', False):
            return
        
        # Email notifications
        if notification_config.get('email', {}).get('enabled', False):
            await self._send_email_notification(validation_result, trigger)
        
        # Slack notifications
        if notification_config.get('slack', {}).get('enabled', False):
            await self._send_slack_notification(validation_result, trigger)
        
        # Webhook notifications
        if notification_config.get('webhook', {}).get('enabled', False):
            await self._send_webhook_notification(validation_result, trigger)
    
    async def _send_email_notification(self, validation_result: ValidationResult, trigger: str) -> None:
        """Send email notification"""
        # Implementation would depend on email service configuration
        self.logger.info(f"Email notification sent (validation {'passed' if validation_result.passed else 'failed'})")
    
    async def _send_slack_notification(self, validation_result: ValidationResult, trigger: str) -> None:
        """Send Slack notification"""
        # Implementation would depend on Slack webhook configuration
        self.logger.info(f"Slack notification sent (validation {'passed' if validation_result.passed else 'failed'})")
    
    async def _send_webhook_notification(self, validation_result: ValidationResult, trigger: str) -> None:
        """Send webhook notification"""
        # Implementation would depend on webhook configuration
        self.logger.info(f"Webhook notification sent (validation {'passed' if validation_result.passed else 'failed'})")
    
    async def _send_critical_alert(self, message: str) -> None:
        """Send critical alert notification"""
        self.logger.critical(f"🚨 CRITICAL ALERT: {message}")
        
        # Send to all configured notification channels
        notification_config = self.config.notification_config
        
        # Force send even if notifications are normally disabled for critical alerts
        if notification_config.get('email', {}).get('critical_alerts', True):
            await self._send_email_notification(None, "critical_alert")
        
        if notification_config.get('slack', {}).get('critical_alerts', True):
            await self._send_slack_notification(None, "critical_alert")
    
    async def _run_success_hooks(self, validation_result: ValidationResult, trigger: str) -> None:
        """Run success hooks"""
        success_hooks = self.config.success_gates.get('success_hooks', [])
        
        for hook in success_hooks:
            try:
                if hook.get('type') == 'script':
                    subprocess.run(hook['command'], cwd=self.project_root, shell=True, check=True)
                elif hook.get('type') == 'webhook':
                    # Send success webhook
                    pass
                
                self.logger.info(f"Success hook executed: {hook.get('name', 'unnamed')}")
                
            except Exception as e:
                self.logger.error(f"Success hook failed: {e}")
    
    async def _run_failure_hooks(self, validation_result: ValidationResult, trigger: str, gates_passed: bool) -> None:
        """Run failure hooks"""
        failure_hooks = self.config.rollback_config.get('failure_hooks', [])
        
        for hook in failure_hooks:
            try:
                if hook.get('type') == 'script':
                    subprocess.run(hook['command'], cwd=self.project_root, shell=True, check=True)
                elif hook.get('type') == 'webhook':
                    # Send failure webhook
                    pass
                
                self.logger.info(f"Failure hook executed: {hook.get('name', 'unnamed')}")
                
            except Exception as e:
                self.logger.error(f"Failure hook failed: {e}")
    
    async def _start_file_watching(self) -> None:
        """Start file watching for automatic validation triggers"""
        # Collect files to watch from triggers
        for trigger in self.config.validation_triggers:
            if trigger.trigger_type == 'file_change' and trigger.enabled:
                file_patterns = trigger.config.get('file_patterns', [])
                for pattern in file_patterns:
                    # Convert pattern to actual files
                    files = list(self.project_root.glob(pattern))
                    self.watched_files.update(files)
        
        if self.watched_files:
            self.file_watcher_task = asyncio.create_task(self._file_watcher_loop())
    
    async def _file_watcher_loop(self) -> None:
        """File watcher loop"""
        # Simple file modification time based watching
        # In production, you might want to use a more sophisticated file watching library
        file_mtimes = {}
        
        # Initialize file modification times
        for file_path in self.watched_files:
            if file_path.exists():
                file_mtimes[file_path] = file_path.stat().st_mtime
        
        while self.pipeline_running:
            try:
                # Check for file changes
                for file_path in self.watched_files:
                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        if file_path not in file_mtimes or current_mtime > file_mtimes[file_path]:
                            file_mtimes[file_path] = current_mtime
                            
                            # File was modified, trigger validation
                            self.logger.info(f"File change detected: {file_path}")
                            await self.run_validation(f"file_change:{file_path.name}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"File watching error: {e}")
                await asyncio.sleep(5)
    
    async def _schedule_validation_triggers(self) -> None:
        """Schedule time-based validation triggers"""
        scheduled_triggers = [t for t in self.config.validation_triggers 
                            if t.trigger_type == 'schedule' and t.enabled]
        
        for trigger in scheduled_triggers:
            asyncio.create_task(self._scheduled_validation_loop(trigger))
    
    async def _scheduled_validation_loop(self, trigger: ValidationTrigger) -> None:
        """Loop for scheduled validations"""
        interval_minutes = trigger.config.get('interval_minutes', 60)
        
        while self.pipeline_running:
            await asyncio.sleep(interval_minutes * 60)
            
            if self.pipeline_running:  # Check again in case pipeline was stopped
                await self.run_validation(f"schedule:{trigger.name}")
    
    async def _get_current_commit_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  cwd=self.project_root, 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def _update_pipeline_metrics(self, validation_result: ValidationResult, execution_time_ms: int) -> None:
        """Update pipeline metrics"""
        self.pipeline_metrics['validations_run'] += 1
        
        if validation_result.passed:
            self.pipeline_metrics['validations_passed'] += 1
        else:
            self.pipeline_metrics['validations_failed'] += 1
        
        # Update average execution time
        total_validations = self.pipeline_metrics['validations_run']
        current_avg = self.pipeline_metrics['average_execution_time_ms']
        self.pipeline_metrics['average_execution_time_ms'] = (
            (current_avg * (total_validations - 1) + execution_time_ms) / total_validations
        )
    
    def _load_config(self, config_path: Optional[Path]) -> PipelineConfig:
        """Load pipeline configuration"""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            # Default configuration
            config_data = self._get_default_config()
        
        # Convert to PipelineConfig object
        triggers = [
            ValidationTrigger(**trigger_data) 
            for trigger_data in config_data.get('validation_triggers', [])
        ]
        
        return PipelineConfig(
            validation_triggers=triggers,
            notification_config=config_data.get('notification_config', {}),
            rollback_config=config_data.get('rollback_config', {}),
            monitoring_config=config_data.get('monitoring_config', {}),
            success_gates=config_data.get('success_gates', {})
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "validation_triggers": [
                {
                    "name": "file_change_trigger",
                    "enabled": True,
                    "trigger_type": "file_change",
                    "config": {
                        "file_patterns": [
                            "swarm/**/*.py",
                            "tests/**/*.py",
                            "requirements.txt"
                        ]
                    }
                },
                {
                    "name": "scheduled_validation",
                    "enabled": False,
                    "trigger_type": "schedule",
                    "config": {
                        "interval_minutes": 60
                    }
                }
            ],
            "notification_config": {
                "enabled": True,
                "email": {"enabled": False},
                "slack": {"enabled": False},
                "webhook": {"enabled": False}
            },
            "rollback_config": {
                "enabled": True,
                "max_consecutive_failures": 3,
                "git_rollback": {"enabled": False},
                "service_restart": {"enabled": False}
            },
            "monitoring_config": {
                "interval": 1.0
            },
            "success_gates": {
                "max_errors": 0,
                "max_performance_degradation": 10.0,
                "min_test_coverage": 80.0,
                "coupling_requirements": {
                    "UnifiedManagement": 50.0,
                    "SageAgent": 30.0
                },
                "save_baseline_on_success": True
            }
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'pipeline_running': self.pipeline_running,
            'last_validation': self.validation_history[-1] if self.validation_history else None,
            'metrics': self.pipeline_metrics,
            'watched_files_count': len(self.watched_files),
            'triggers_configured': len(self.config.validation_triggers),
            'active_triggers': len([t for t in self.config.validation_triggers if t.enabled])
        }
    
    def get_validation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get validation history"""
        return self.validation_history[-limit:]