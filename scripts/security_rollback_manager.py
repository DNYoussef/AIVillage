#!/usr/bin/env python3
"""
Security Rollback Manager

Provides rollback capabilities for security fixes to ensure zero-downtime
recovery in case of issues during security remediation deployment.

Features:
- Pre-deployment state capture
- Incremental rollback capabilities
- Validation before rollback execution
- Rollback verification and testing
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityRollbackManager:
    """Manages rollback operations for security fixes"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getcwd())
        self.rollback_dir = self.base_path / "rollbacks" / "security"
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        
    def create_rollback_point(self, name: str, description: str = "") -> str:
        """Create a rollback point before applying security fixes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rollback_id = f"{name}_{timestamp}"
        rollback_path = self.rollback_dir / rollback_id
        rollback_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating rollback point: {rollback_id}")
        
        # Capture current git state
        self._capture_git_state(rollback_path)
        
        # Capture environment variables
        self._capture_environment_state(rollback_path)
        
        # Capture configuration files
        self._capture_configuration_state(rollback_path)
        
        # Capture security-related files
        self._capture_security_files(rollback_path)
        
        # Create rollback manifest
        self._create_rollback_manifest(rollback_path, rollback_id, description)
        
        logger.info(f"Rollback point created: {rollback_path}")
        return rollback_id
        
    def _capture_git_state(self, rollback_path: Path):
        """Capture current git state for rollback"""
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.base_path
            )
            if result.returncode == 0:
                with open(rollback_path / "git_commit.txt", "w") as f:
                    f.write(result.stdout.strip())
            
            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=self.base_path
            )
            if result.returncode == 0:
                with open(rollback_path / "git_branch.txt", "w") as f:
                    f.write(result.stdout.strip())
                    
            # Get git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.base_path
            )
            if result.returncode == 0:
                with open(rollback_path / "git_status.txt", "w") as f:
                    f.write(result.stdout)
                    
        except Exception as e:
            logger.warning(f"Failed to capture git state: {e}")
            
    def _capture_environment_state(self, rollback_path: Path):
        """Capture environment variables relevant to security"""
        security_env_vars = [
            "API_KEY", "JWT_SECRET", "DATABASE_URL", "REDIS_URL",
            "SMTP_HOST", "SMTP_USER", "ENVIRONMENT", "DEBUG"
        ]
        
        env_state = {}
        for var in security_env_vars:
            value = os.environ.get(var)
            if value:
                # Don't store actual secrets, just indicate presence
                env_state[var] = "***SET***" if any(secret in var.lower() for secret in ["key", "secret", "password", "token"]) else value
            else:
                env_state[var] = "***NOT_SET***"
        
        with open(rollback_path / "environment_state.json", "w") as f:
            json.dump(env_state, f, indent=2)
            
    def _capture_configuration_state(self, rollback_path: Path):
        """Capture configuration files state"""
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.conf"
        ]
        
        config_files = []
        for pattern in config_patterns:
            config_files.extend(self.base_path.rglob(pattern))
        
        # Filter to important config files
        important_configs = [
            f for f in config_files 
            if any(path_part in str(f).lower() for path_part in [
                "config", "setting", "env", "docker", "security", "auth"
            ]) and "node_modules" not in str(f) and ".git" not in str(f)
        ]
        
        config_backup_dir = rollback_path / "configurations"
        config_backup_dir.mkdir(exist_ok=True)
        
        backed_up_configs = []
        for config_file in important_configs[:20]:  # Limit to avoid excessive backup
            try:
                relative_path = config_file.relative_to(self.base_path)
                backup_path = config_backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, backup_path)
                backed_up_configs.append(str(relative_path))
            except Exception as e:
                logger.warning(f"Failed to backup config {config_file}: {e}")
        
        # Record what was backed up
        with open(rollback_path / "backed_up_configs.json", "w") as f:
            json.dump(backed_up_configs, f, indent=2)
            
    def _capture_security_files(self, rollback_path: Path):
        """Capture current state of security-related files"""
        security_dirs = [
            "security", "config/security", "reports/security", 
            "tests/security", "scripts/security"
        ]
        
        security_backup_dir = rollback_path / "security_files"
        security_backup_dir.mkdir(exist_ok=True)
        
        backed_up_files = []
        for sec_dir in security_dirs:
            sec_path = self.base_path / sec_dir
            if sec_path.exists():
                try:
                    backup_sec_path = security_backup_dir / sec_dir
                    backup_sec_path.parent.mkdir(parents=True, exist_ok=True)
                    if sec_path.is_dir():
                        shutil.copytree(sec_path, backup_sec_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(sec_path, backup_sec_path)
                    backed_up_files.append(sec_dir)
                except Exception as e:
                    logger.warning(f"Failed to backup security dir {sec_dir}: {e}")
        
        with open(rollback_path / "backed_up_security.json", "w") as f:
            json.dump(backed_up_files, f, indent=2)
            
    def _create_rollback_manifest(self, rollback_path: Path, rollback_id: str, description: str):
        """Create manifest describing the rollback point"""
        manifest = {
            "rollback_id": rollback_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "base_path": str(self.base_path),
            "components": {
                "git_state": "Captured current commit, branch, and status",
                "environment_state": "Captured security-relevant environment variables",
                "configurations": "Backed up configuration files",
                "security_files": "Backed up security-related directories"
            },
            "rollback_steps": [
                "1. Verify current system state",
                "2. Restore git commit if needed",
                "3. Restore configuration files",
                "4. Restore security files",
                "5. Validate rollback success",
                "6. Restart affected services"
            ]
        }
        
        with open(rollback_path / "rollback_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
    def list_rollback_points(self) -> List[Dict[str, Any]]:
        """List available rollback points"""
        rollback_points = []
        
        if not self.rollback_dir.exists():
            return rollback_points
            
        for rollback_path in self.rollback_dir.iterdir():
            if rollback_path.is_dir():
                manifest_file = rollback_path / "rollback_manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r") as f:
                            manifest = json.load(f)
                        rollback_points.append(manifest)
                    except Exception as e:
                        logger.warning(f"Failed to read manifest for {rollback_path}: {e}")
        
        return sorted(rollback_points, key=lambda x: x["created_at"], reverse=True)
        
    def execute_rollback(self, rollback_id: str, validate: bool = True) -> bool:
        """Execute rollback to a specific point"""
        rollback_path = self.rollback_dir / rollback_id
        
        if not rollback_path.exists():
            logger.error(f"Rollback point not found: {rollback_id}")
            return False
        
        manifest_file = rollback_path / "rollback_manifest.json"
        if not manifest_file.exists():
            logger.error(f"Rollback manifest not found: {manifest_file}")
            return False
            
        try:
            with open(manifest_file, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read rollback manifest: {e}")
            return False
        
        logger.info(f"Starting rollback to: {rollback_id}")
        logger.info(f"Description: {manifest.get('description', 'No description')}")
        
        # Validation before rollback
        if validate and not self._validate_rollback_safety(rollback_path):
            logger.error("Rollback safety validation failed")
            return False
        
        try:
            # Step 1: Create backup of current state (rollback of rollback)
            current_backup_id = self.create_rollback_point(
                f"pre_rollback_{rollback_id}", 
                "Backup before executing rollback"
            )
            logger.info(f"Created safety backup: {current_backup_id}")
            
            # Step 2: Restore git state if needed
            if not self._restore_git_state(rollback_path):
                logger.warning("Git state restoration failed, continuing...")
            
            # Step 3: Restore configuration files
            if not self._restore_configurations(rollback_path):
                logger.warning("Configuration restoration failed, continuing...")
                
            # Step 4: Restore security files
            if not self._restore_security_files(rollback_path):
                logger.warning("Security files restoration failed, continuing...")
            
            # Step 5: Validate rollback success
            if validate and not self._validate_rollback_success(rollback_path):
                logger.error("Rollback validation failed")
                return False
            
            logger.info(f"Rollback to {rollback_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False
            
    def _validate_rollback_safety(self, rollback_path: Path) -> bool:
        """Validate that rollback is safe to execute"""
        try:
            # Check if rollback files exist
            required_files = ["rollback_manifest.json"]
            for req_file in required_files:
                if not (rollback_path / req_file).exists():
                    logger.error(f"Required rollback file missing: {req_file}")
                    return False
            
            # Check git repository state
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.base_path
            )
            
            if result.returncode == 0 and result.stdout.strip():
                logger.warning("Git repository has uncommitted changes")
                response = input("Continue with rollback? (y/N): ")
                if response.lower() != 'y':
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback safety validation failed: {e}")
            return False
            
    def _restore_git_state(self, rollback_path: Path) -> bool:
        """Restore git state from rollback point"""
        try:
            commit_file = rollback_path / "git_commit.txt"
            if commit_file.exists():
                with open(commit_file, "r") as f:
                    target_commit = f.read().strip()
                
                result = subprocess.run(
                    ["git", "checkout", target_commit],
                    capture_output=True, text=True, cwd=self.base_path
                )
                
                if result.returncode == 0:
                    logger.info(f"Restored git state to commit: {target_commit}")
                    return True
                else:
                    logger.error(f"Failed to checkout commit {target_commit}: {result.stderr}")
                    
            return False
            
        except Exception as e:
            logger.error(f"Failed to restore git state: {e}")
            return False
            
    def _restore_configurations(self, rollback_path: Path) -> bool:
        """Restore configuration files from rollback point"""
        try:
            config_backup_dir = rollback_path / "configurations"
            backed_up_configs_file = rollback_path / "backed_up_configs.json"
            
            if not config_backup_dir.exists() or not backed_up_configs_file.exists():
                logger.info("No configuration backups to restore")
                return True
            
            with open(backed_up_configs_file, "r") as f:
                backed_up_configs = json.load(f)
            
            restored_count = 0
            for config_path in backed_up_configs:
                backup_file = config_backup_dir / config_path
                target_file = self.base_path / config_path
                
                if backup_file.exists():
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_file, target_file)
                    restored_count += 1
            
            logger.info(f"Restored {restored_count} configuration files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore configurations: {e}")
            return False
            
    def _restore_security_files(self, rollback_path: Path) -> bool:
        """Restore security files from rollback point"""
        try:
            security_backup_dir = rollback_path / "security_files"
            backed_up_security_file = rollback_path / "backed_up_security.json"
            
            if not security_backup_dir.exists() or not backed_up_security_file.exists():
                logger.info("No security file backups to restore")
                return True
            
            with open(backed_up_security_file, "r") as f:
                backed_up_security = json.load(f)
            
            restored_count = 0
            for sec_path in backed_up_security:
                backup_path = security_backup_dir / sec_path
                target_path = self.base_path / sec_path
                
                if backup_path.exists():
                    if target_path.exists():
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                    
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if backup_path.is_dir():
                        shutil.copytree(backup_path, target_path)
                    else:
                        shutil.copy2(backup_path, target_path)
                    restored_count += 1
            
            logger.info(f"Restored {restored_count} security file locations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore security files: {e}")
            return False
            
    def _validate_rollback_success(self, rollback_path: Path) -> bool:
        """Validate that rollback was successful"""
        try:
            # Run basic validation tests
            logger.info("Validating rollback success...")
            
            # Test that core modules can still be imported
            test_modules = ["core.agent_forge", "core.rag"]
            for module in test_modules:
                result = subprocess.run([
                    sys.executable, "-c", f"import {module}; print('OK')"
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0 or "OK" not in result.stdout:
                    logger.error(f"Module {module} failed to import after rollback")
                    return False
            
            logger.info("Rollback validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Rollback validation failed: {e}")
            return False
            
    def cleanup_old_rollbacks(self, keep_count: int = 10):
        """Clean up old rollback points, keeping only the most recent ones"""
        rollback_points = self.list_rollback_points()
        
        if len(rollback_points) <= keep_count:
            logger.info(f"Only {len(rollback_points)} rollback points, no cleanup needed")
            return
        
        to_remove = rollback_points[keep_count:]
        removed_count = 0
        
        for rollback in to_remove:
            rollback_path = self.rollback_dir / rollback["rollback_id"]
            try:
                shutil.rmtree(rollback_path)
                removed_count += 1
                logger.info(f"Removed old rollback: {rollback['rollback_id']}")
            except Exception as e:
                logger.warning(f"Failed to remove rollback {rollback['rollback_id']}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old rollback points")

def main():
    """Main CLI interface for rollback manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Rollback Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create rollback point
    create_parser = subparsers.add_parser("create", help="Create a rollback point")
    create_parser.add_argument("name", help="Name for the rollback point")
    create_parser.add_argument("-d", "--description", default="", help="Description of the rollback point")
    
    # List rollback points
    list_parser = subparsers.add_parser("list", help="List available rollback points")
    
    # Execute rollback
    rollback_parser = subparsers.add_parser("rollback", help="Execute rollback to a point")
    rollback_parser.add_argument("rollback_id", help="ID of the rollback point")
    rollback_parser.add_argument("--no-validate", action="store_true", help="Skip validation steps")
    
    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old rollback points")
    cleanup_parser.add_argument("-k", "--keep", type=int, default=10, help="Number of rollback points to keep")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = SecurityRollbackManager()
    
    if args.command == "create":
        rollback_id = manager.create_rollback_point(args.name, args.description)
        print(f"Created rollback point: {rollback_id}")
        
    elif args.command == "list":
        rollback_points = manager.list_rollback_points()
        if not rollback_points:
            print("No rollback points found")
        else:
            print(f"{'ID':<30} {'Created':<20} {'Description'}")
            print("-" * 70)
            for rp in rollback_points:
                created = rp["created_at"][:19].replace("T", " ")
                desc = rp.get("description", "")[:30]
                print(f"{rp['rollback_id']:<30} {created:<20} {desc}")
                
    elif args.command == "rollback":
        success = manager.execute_rollback(args.rollback_id, validate=not args.no_validate)
        if success:
            print("Rollback executed successfully")
        else:
            print("Rollback failed")
            sys.exit(1)
            
    elif args.command == "cleanup":
        manager.cleanup_old_rollbacks(args.keep)

if __name__ == "__main__":
    main()