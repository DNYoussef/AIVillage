"""
Configuration Validation and Hot-Reload System
Provides real-time configuration validation and hot-reload capabilities
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from threading import RLock

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """Represents a configuration validation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # schema, constraint, security, performance
    priority: int   # 1=critical, 2=high, 3=medium, 4=low
    pattern: Optional[str] = None
    validator_function: Optional[str] = None
    error_message: str = ""
    fix_suggestion: str = ""

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    file_path: str
    rule_id: str
    rule_name: str
    passed: bool
    severity: str
    message: str
    fix_suggestion: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None

@dataclass
class ConfigurationChange:
    """Represents a configuration file change"""
    file_path: str
    change_type: str  # modified, created, deleted, moved
    timestamp: datetime
    old_checksum: Optional[str] = None
    new_checksum: Optional[str] = None
    validation_results: List[ValidationResult] = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []

class ConfigurationFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes"""
    
    def __init__(self, validation_system: 'ConfigurationValidationSystem'):
        self.validation_system = validation_system
        
    def on_modified(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.validation_system.handle_file_change(event.src_path, "modified")
            )
            
    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.validation_system.handle_file_change(event.src_path, "created")
            )
            
    def on_deleted(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.validation_system.handle_file_change(event.src_path, "deleted")
            )
            
    def on_moved(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.validation_system.handle_file_change(event.dest_path, "moved")
            )

class ConfigurationValidationSystem:
    """Main configuration validation and hot-reload system"""
    
    def __init__(self, config_directories: List[str]):
        self.config_directories = [Path(d) for d in config_directories]
        
        # Validation rules
        self._validation_rules: Dict[str, ValidationRule] = {}
        self._load_default_validation_rules()
        
        # File watching
        self._file_observer = Observer()
        self._file_watcher = ConfigurationFileWatcher(self)
        self._watched_files: Dict[str, str] = {}  # file_path -> checksum
        
        # Hot-reload callbacks
        self._reload_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Change tracking
        self._recent_changes: List[ConfigurationChange] = []
        self._change_lock = RLock()
        
        # Validation cache
        self._validation_cache: Dict[str, List[ValidationResult]] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        # Configuration snapshots for rollback
        self._config_snapshots: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the validation and hot-reload system"""
        logger.info("Initializing Configuration Validation and Hot-Reload System")
        
        # Setup file watching
        await self._setup_file_watching()
        
        # Initial validation of all configuration files
        await self._validate_all_configurations()
        
        # Start change processing loop
        asyncio.create_task(self._process_changes_loop())
        
        logger.info("Configuration validation system initialized")
        
    async def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule"""
        self._validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.rule_id}")
        
    async def register_reload_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback for configuration hot-reload"""
        self._reload_callbacks.append(callback)
        logger.info("Registered hot-reload callback")
        
    async def validate_configuration_file(self, file_path: str, use_cache: bool = True) -> List[ValidationResult]:
        """Validate a specific configuration file"""
        
        file_path = str(Path(file_path).resolve())
        
        # Check cache first
        if use_cache and file_path in self._validation_cache:
            cache_expiry = self._cache_expiry.get(file_path, datetime.min)
            if datetime.now() < cache_expiry:
                return self._validation_cache[file_path]
                
        results = []
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                results.append(ValidationResult(
                    file_path=file_path,
                    rule_id="file_existence",
                    rule_name="File Existence Check",
                    passed=False,
                    severity="critical",
                    message=f"Configuration file does not exist: {file_path}",
                    fix_suggestion="Create the missing configuration file"
                ))
                return results
                
            # Load and parse configuration
            config_data = await self._load_configuration_file(file_path)
            if config_data is None:
                results.append(ValidationResult(
                    file_path=file_path,
                    rule_id="file_parsing",
                    rule_name="File Parsing Check",
                    passed=False,
                    severity="critical",
                    message=f"Failed to parse configuration file: {file_path}",
                    fix_suggestion="Check file syntax and format"
                ))
                return results
                
            # Run validation rules
            for rule_id, rule in self._validation_rules.items():
                validation_result = await self._apply_validation_rule(file_path, config_data, rule)
                if validation_result:
                    results.append(validation_result)
                    
            # Cache results for 5 minutes
            self._validation_cache[file_path] = results
            self._cache_expiry[file_path] = datetime.now() + timedelta(minutes=5)
            
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            results.append(ValidationResult(
                file_path=file_path,
                rule_id="validation_error",
                rule_name="Validation Error",
                passed=False,
                severity="high",
                message=f"Validation error: {e}",
                fix_suggestion="Check file format and content"
            ))
            
        return results
        
    async def validate_all_configurations(self) -> Dict[str, List[ValidationResult]]:
        """Validate all configuration files"""
        
        all_results = {}
        
        for config_dir in self.config_directories:
            if config_dir.exists():
                for config_file in config_dir.rglob("*"):
                    if config_file.is_file() and self._is_config_file(config_file):
                        file_path = str(config_file)
                        results = await self.validate_configuration_file(file_path, use_cache=False)
                        all_results[file_path] = results
                        
        return all_results
        
    async def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary across all configurations"""
        
        all_results = await self.validate_all_configurations()
        
        total_files = len(all_results)
        total_issues = sum(len(results) for results in all_results.values())
        
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        files_with_issues = 0
        
        for file_path, results in all_results.items():
            if results:
                files_with_issues += 1
                for result in results:
                    if not result.passed:
                        severity_counts[result.severity] += 1
                        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "files_with_issues": files_with_issues,
            "total_issues": total_issues,
            "issues_by_severity": severity_counts,
            "files_clean": total_files - files_with_issues,
            "validation_rules_active": len(self._validation_rules),
            "recent_changes": len(self._recent_changes)
        }
        
    async def create_configuration_snapshot(self, snapshot_name: str):
        """Create a snapshot of current configuration state"""
        
        snapshot = {}
        
        for config_dir in self.config_directories:
            if config_dir.exists():
                for config_file in config_dir.rglob("*"):
                    if config_file.is_file() and self._is_config_file(config_file):
                        file_path = str(config_file)
                        config_data = await self._load_configuration_file(file_path)
                        if config_data is not None:
                            snapshot[file_path] = config_data
                            
        self._config_snapshots[snapshot_name] = snapshot
        logger.info(f"Created configuration snapshot: {snapshot_name}")
        
    async def rollback_to_snapshot(self, snapshot_name: str) -> bool:
        """Rollback configuration to a previous snapshot"""
        
        if snapshot_name not in self._config_snapshots:
            logger.error(f"Snapshot not found: {snapshot_name}")
            return False
            
        snapshot = self._config_snapshots[snapshot_name]
        
        try:
            for file_path, config_data in snapshot.items():
                await self._save_configuration_file(file_path, config_data)
                
            logger.info(f"Rolled back to snapshot: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
            
    async def handle_file_change(self, file_path: str, change_type: str):
        """Handle configuration file changes"""
        
        file_path = str(Path(file_path).resolve())
        
        # Skip if not a configuration file
        if not self._is_config_file(Path(file_path)):
            return
            
        old_checksum = self._watched_files.get(file_path)
        new_checksum = await self._calculate_file_checksum(file_path)
        
        # Skip if file hasn't actually changed
        if change_type == "modified" and old_checksum == new_checksum:
            return
            
        # Create change record
        change = ConfigurationChange(
            file_path=file_path,
            change_type=change_type,
            timestamp=datetime.now(),
            old_checksum=old_checksum,
            new_checksum=new_checksum
        )
        
        # Validate the changed file
        if change_type != "deleted":
            validation_results = await self.validate_configuration_file(file_path, use_cache=False)
            change.validation_results = validation_results
            
            # Log validation results
            critical_issues = [r for r in validation_results if r.severity == "critical" and not r.passed]
            if critical_issues:
                logger.error(f"Critical validation issues in {file_path}:")
                for issue in critical_issues:
                    logger.error(f"  - {issue.message}")
                    
        # Update file checksum tracking
        if change_type == "deleted":
            self._watched_files.pop(file_path, None)
        else:
            self._watched_files[file_path] = new_checksum
            
        # Add to recent changes
        with self._change_lock:
            self._recent_changes.append(change)
            # Keep only last 100 changes
            self._recent_changes = self._recent_changes[-100:]
            
        # Trigger hot-reload if validation passes
        if change_type != "deleted" and not critical_issues:
            await self._trigger_hot_reload(file_path)
            
        logger.info(f"Processed configuration change: {file_path} ({change_type})")
        
    async def get_recent_changes(self, limit: int = 10) -> List[ConfigurationChange]:
        """Get recent configuration changes"""
        with self._change_lock:
            return self._recent_changes[-limit:]
            
    async def _setup_file_watching(self):
        """Setup file system watching for configuration directories"""
        
        for config_dir in self.config_directories:
            if config_dir.exists():
                self._file_observer.schedule(
                    self._file_watcher, 
                    str(config_dir), 
                    recursive=True
                )
                
                # Initialize file checksums
                for config_file in config_dir.rglob("*"):
                    if config_file.is_file() and self._is_config_file(config_file):
                        file_path = str(config_file)
                        checksum = await self._calculate_file_checksum(file_path)
                        self._watched_files[file_path] = checksum
                        
        self._file_observer.start()
        logger.info("File watching started for configuration directories")
        
    async def _validate_all_configurations(self):
        """Initial validation of all configuration files"""
        
        validation_results = await self.validate_all_configurations()
        
        total_issues = 0
        critical_issues = 0
        
        for file_path, results in validation_results.items():
            for result in results:
                if not result.passed:
                    total_issues += 1
                    if result.severity == "critical":
                        critical_issues += 1
                        
        logger.info(f"Initial validation completed: {total_issues} issues found ({critical_issues} critical)")
        
    async def _process_changes_loop(self):
        """Background loop to process configuration changes"""
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Clean up old validation cache entries
                now = datetime.now()
                expired_cache_keys = [
                    key for key, expiry in self._cache_expiry.items()
                    if now > expiry
                ]
                
                for key in expired_cache_keys:
                    self._validation_cache.pop(key, None)
                    self._cache_expiry.pop(key, None)
                    
            except Exception as e:
                logger.error(f"Change processing loop error: {e}")
                
    def _load_default_validation_rules(self):
        """Load default validation rules"""
        
        # Schema validation rules
        self._validation_rules["yaml_syntax"] = ValidationRule(
            rule_id="yaml_syntax",
            name="YAML Syntax Check",
            description="Validate YAML file syntax",
            rule_type="schema",
            priority=1,
            error_message="Invalid YAML syntax",
            fix_suggestion="Check YAML indentation and structure"
        )
        
        # Security validation rules
        self._validation_rules["no_hardcoded_secrets"] = ValidationRule(
            rule_id="no_hardcoded_secrets",
            name="No Hardcoded Secrets",
            description="Check for hardcoded passwords and secrets",
            rule_type="security",
            priority=1,
            pattern=r"(password|secret|key|token)\s*[:=]\s*['\"][^'\"]+['\"]",
            error_message="Hardcoded secret detected",
            fix_suggestion="Use environment variables or secure vaults for secrets"
        )
        
        self._validation_rules["secure_protocols"] = ValidationRule(
            rule_id="secure_protocols",
            name="Secure Protocols Only",
            description="Ensure only secure protocols are used",
            rule_type="security",
            priority=2,
            pattern=r"http://|ftp://|telnet://",
            error_message="Insecure protocol detected",
            fix_suggestion="Use HTTPS, SFTP, or SSH instead"
        )
        
        # Performance validation rules
        self._validation_rules["reasonable_timeouts"] = ValidationRule(
            rule_id="reasonable_timeouts",
            name="Reasonable Timeouts",
            description="Check for reasonable timeout values",
            rule_type="performance",
            priority=3,
            error_message="Unreasonable timeout value",
            fix_suggestion="Use reasonable timeout values (1-300 seconds)"
        )
        
        # Configuration consistency rules
        self._validation_rules["unique_ports"] = ValidationRule(
            rule_id="unique_ports",
            name="Unique Port Numbers",
            description="Ensure port numbers are unique across services",
            rule_type="constraint",
            priority=2,
            error_message="Port number conflict detected",
            fix_suggestion="Assign unique port numbers to each service"
        )
        
    async def _load_configuration_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load and parse configuration file"""
        
        try:
            path = Path(file_path)
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif path.suffix == '.json':
                return json.loads(content)
            elif '.env' in path.name:
                # Simple env file parsing
                env_config = {}
                for line in content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        env_config[key.strip()] = value.strip()
                return env_config
            else:
                logger.warning(f"Unsupported configuration format: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            return None
            
    async def _save_configuration_file(self, file_path: str, config_data: Dict[str, Any]):
        """Save configuration data to file"""
        
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
                elif path.suffix == '.json':
                    json.dump(config_data, f, indent=2, sort_keys=True)
                else:
                    logger.error(f"Cannot save unsupported format: {file_path}")
                    return
                    
            logger.info(f"Saved configuration file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration file {file_path}: {e}")
            raise
            
    async def _apply_validation_rule(self, file_path: str, config_data: Dict[str, Any], rule: ValidationRule) -> Optional[ValidationResult]:
        """Apply a validation rule to configuration data"""
        
        try:
            if rule.rule_id == "yaml_syntax":
                # YAML syntax is already checked during loading
                return None
                
            elif rule.rule_id == "no_hardcoded_secrets":
                return await self._check_hardcoded_secrets(file_path, config_data, rule)
                
            elif rule.rule_id == "secure_protocols":
                return await self._check_secure_protocols(file_path, config_data, rule)
                
            elif rule.rule_id == "reasonable_timeouts":
                return await self._check_reasonable_timeouts(file_path, config_data, rule)
                
            elif rule.rule_id == "unique_ports":
                # This is handled at system level, not per file
                return None
                
            else:
                logger.warning(f"Unknown validation rule: {rule.rule_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying validation rule {rule.rule_id} to {file_path}: {e}")
            return None
            
    async def _check_hardcoded_secrets(self, file_path: str, config_data: Dict[str, Any], rule: ValidationRule) -> Optional[ValidationResult]:
        """Check for hardcoded secrets in configuration"""
        
        sensitive_keys = ["password", "secret", "key", "token", "api_key", "private_key"]
        
        def check_value(key: str, value: Any, path: str = "") -> Optional[str]:
            if isinstance(value, str):
                # Check if key indicates a secret and value looks hardcoded
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    # Skip if value is a placeholder or environment variable
                    if not (value.startswith("${") or value.startswith("$") or 
                           value in ["", "TODO", "CHANGEME", "your-secret-here"]):
                        return f"{path}.{key}" if path else key
            elif isinstance(value, dict):
                for k, v in value.items():
                    new_path = f"{path}.{k}" if path else k
                    result = check_value(k, v, new_path)
                    if result:
                        return result
            return None
            
        secret_location = check_value("", config_data)
        
        if secret_location:
            return ValidationResult(
                file_path=file_path,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                severity="critical",
                message=f"Hardcoded secret detected at: {secret_location}",
                fix_suggestion=rule.fix_suggestion
            )
            
        return None
        
    async def _check_secure_protocols(self, file_path: str, config_data: Dict[str, Any], rule: ValidationRule) -> Optional[ValidationResult]:
        """Check for insecure protocols in configuration"""
        
        insecure_protocols = ["http://", "ftp://", "telnet://"]
        
        def check_protocols(data: Any, path: str = "") -> Optional[str]:
            if isinstance(data, str):
                for protocol in insecure_protocols:
                    if protocol in data.lower():
                        return path or "root"
            elif isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    result = check_protocols(value, new_path)
                    if result:
                        return result
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    result = check_protocols(item, new_path)
                    if result:
                        return result
            return None
            
        insecure_location = check_protocols(config_data)
        
        if insecure_location:
            return ValidationResult(
                file_path=file_path,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                severity="high",
                message=f"Insecure protocol detected at: {insecure_location}",
                fix_suggestion=rule.fix_suggestion
            )
            
        return None
        
    async def _check_reasonable_timeouts(self, file_path: str, config_data: Dict[str, Any], rule: ValidationRule) -> Optional[ValidationResult]:
        """Check for reasonable timeout values"""
        
        def check_timeouts(data: Any, path: str = "") -> Optional[Tuple[str, Any]]:
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    
                    # Check if key indicates a timeout
                    if "timeout" in key.lower():
                        if isinstance(value, (int, float)):
                            # Check if timeout is unreasonable (< 1 second or > 5 minutes)
                            if value < 1 or value > 300:
                                return (new_path, value)
                        elif isinstance(value, str) and value.isdigit():
                            timeout_val = float(value)
                            if timeout_val < 1 or timeout_val > 300:
                                return (new_path, timeout_val)
                                
                    # Recurse into nested structures
                    result = check_timeouts(value, new_path)
                    if result:
                        return result
                        
            return None
            
        timeout_issue = check_timeouts(config_data)
        
        if timeout_issue:
            location, value = timeout_issue
            return ValidationResult(
                file_path=file_path,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                passed=False,
                severity="medium",
                message=f"Unreasonable timeout value {value} at: {location}",
                fix_suggestion=rule.fix_suggestion
            )
            
        return None
        
    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file"""
        
        config_extensions = ['.yaml', '.yml', '.json']
        config_patterns = ['config', '.env']
        
        # Check extension
        if file_path.suffix in config_extensions:
            return True
            
        # Check filename patterns
        filename_lower = file_path.name.lower()
        if any(pattern in filename_lower for pattern in config_patterns):
            return True
            
        return False
        
    async def _calculate_file_checksum(self, file_path: str) -> Optional[str]:
        """Calculate checksum for file"""
        
        try:
            if not Path(file_path).exists():
                return None
                
            with open(file_path, 'rb') as f:
                content = f.read()
                
            return hashlib.sha256(content).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return None
            
    async def _trigger_hot_reload(self, file_path: str):
        """Trigger hot-reload for configuration changes"""
        
        try:
            # Load the updated configuration
            config_data = await self._load_configuration_file(file_path)
            if config_data is None:
                return
                
            # Notify all registered callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(file_path, config_data)
                except Exception as e:
                    logger.error(f"Hot-reload callback error: {e}")
                    
            logger.info(f"Hot-reload triggered for: {file_path}")
            
        except Exception as e:
            logger.error(f"Hot-reload failed for {file_path}: {e}")

# Factory function
async def create_validation_system(config_directories: List[str]) -> ConfigurationValidationSystem:
    """Create and initialize configuration validation system"""
    
    system = ConfigurationValidationSystem(config_directories)
    await system.initialize()
    return system

if __name__ == "__main__":
    async def test_validation_system():
        config_dirs = ["config", "tools/ci-cd/deployment/k8s"]
        
        system = await create_validation_system(config_dirs)
        
        # Test validation
        summary = await system.get_validation_summary()
        print(f"Validation Summary: {summary}")
        
        # Create snapshot
        await system.create_configuration_snapshot("test_snapshot")
        
        # Register hot-reload callback
        def on_config_change(file_path: str, config_data: Dict[str, Any]):
            print(f"Configuration changed: {file_path}")
            
        await system.register_reload_callback(on_config_change)
        
        # Keep running to test file watching
        await asyncio.sleep(60)
        
    asyncio.run(test_validation_system())