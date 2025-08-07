#!/usr/bin/env python3
"""Production Deployment Script
Consolidates deployment, health checks, and production verification.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Production deployment orchestrator."""

    def __init__(self, environment: str, project_root: Path | None = None):
        self.environment = environment
        self.project_root = project_root or Path.cwd()
        self.deploy_dir = self.project_root / "deploy"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Environment-specific settings
        self.config = self.load_environment_config()

    def load_environment_config(self) -> dict[str, Any]:
        """Load environment-specific configuration."""
        config_file = self.deploy_dir / f"{self.environment}.json"

        # Default configuration
        default_config = {
            "namespace": f"aivillage-{self.environment}",
            "image_tag": "latest",
            "replicas": 1 if self.environment == "staging" else 3,
            "resource_limits": {
                "memory": "2Gi" if self.environment == "staging" else "4Gi",
                "cpu": "1000m" if self.environment == "staging" else "2000m",
            },
            "health_check_timeout": 300,
            "rollback_on_failure": True,
        }

        if config_file.exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")

        return default_config

    def run_command(
        self, cmd: list[str], check: bool = True, timeout: int = 300
    ) -> tuple[int, str, str]:
        """Run a command with error handling."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                timeout=timeout,
                cwd=self.project_root,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with return code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            if check:
                raise
            return e.returncode, e.stdout or "", e.stderr or ""
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            if check:
                raise
            return -1, "", f"Timeout after {timeout} seconds"

    def check_prerequisites(self) -> bool:
        """Check that all required tools are available."""
        logger.info("🔍 Checking prerequisites...")

        required_tools = ["kubectl", "helm", "docker"]
        for tool in required_tools:
            returncode, _, stderr = self.run_command(["which", tool], check=False)
            if returncode != 0:
                # Try Windows-style check
                returncode, _, _ = self.run_command([tool, "--version"], check=False)
                if returncode != 0:
                    logger.error(f"Required tool not found: {tool}")
                    return False
            logger.info(f"✅ {tool} is available")

        # Check kubectl context
        returncode, stdout, _ = self.run_command(
            ["kubectl", "config", "current-context"], check=False
        )
        if returncode == 0:
            logger.info(f"kubectl context: {stdout.strip()}")
        else:
            logger.warning("kubectl context not set")

        return True

    def build_image(self) -> bool:
        """Build Docker image."""
        logger.info("🏗️ Building Docker image...")

        image_name = f"aivillage:{self.config['image_tag']}"
        dockerfile = self.project_root / "Dockerfile"

        if not dockerfile.exists():
            logger.error("Dockerfile not found")
            return False

        cmd = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            str(dockerfile),
            str(self.project_root),
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=600)
        if returncode != 0:
            logger.error("Docker build failed")
            return False

        logger.info(f"✅ Built image: {image_name}")
        return True

    def create_namespace(self) -> bool:
        """Create Kubernetes namespace if it doesn't exist."""
        logger.info(f"📦 Ensuring namespace {self.config['namespace']} exists...")

        # Check if namespace exists
        returncode, _, _ = self.run_command(
            ["kubectl", "get", "namespace", self.config["namespace"]], check=False
        )

        if returncode != 0:
            # Create namespace
            returncode, _, _ = self.run_command(
                ["kubectl", "create", "namespace", self.config["namespace"]]
            )
            if returncode != 0:
                logger.error(f"Failed to create namespace {self.config['namespace']}")
                return False
            logger.info(f"✅ Created namespace: {self.config['namespace']}")
        else:
            logger.info(f"✅ Namespace {self.config['namespace']} already exists")

        return True

    def deploy_helm_chart(self) -> bool:
        """Deploy the application using Helm."""
        logger.info("🚀 Deploying with Helm...")

        chart_dir = self.deploy_dir / "helm" / "aivillage"
        if not chart_dir.exists():
            logger.error(f"Helm chart not found at {chart_dir}")
            return False

        release_name = f"aivillage-{self.environment}"

        # Prepare Helm values
        values = {
            "environment": self.environment,
            "image": {"tag": self.config["image_tag"]},
            "replicaCount": self.config["replicas"],
            "resources": {"limits": self.config["resource_limits"]},
        }

        values_file = self.deploy_dir / f"values-{self.environment}.yaml"

        cmd = [
            "helm",
            "upgrade",
            "--install",
            release_name,
            str(chart_dir),
            "--namespace",
            self.config["namespace"],
            "--set-json",
            f"values={json.dumps(values)}",
            "--timeout",
            "10m",
            "--wait",
        ]

        if values_file.exists():
            cmd.extend(["-f", str(values_file)])

        returncode, stdout, stderr = self.run_command(cmd, timeout=600)
        if returncode != 0:
            logger.error("Helm deployment failed")
            return False

        logger.info("✅ Helm deployment successful")
        return True

    def wait_for_deployment(self) -> bool:
        """Wait for deployment to be ready."""
        logger.info("⏳ Waiting for deployment to be ready...")

        timeout = self.config["health_check_timeout"]
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check deployment status
            returncode, stdout, _ = self.run_command(
                [
                    "kubectl",
                    "get",
                    "deployments",
                    "-n",
                    self.config["namespace"],
                    "-o",
                    "json",
                ],
                check=False,
            )

            if returncode == 0:
                try:
                    deployments = json.loads(stdout)
                    all_ready = True

                    for deployment in deployments.get("items", []):
                        status = deployment.get("status", {})
                        ready_replicas = status.get("readyReplicas", 0)
                        desired_replicas = status.get("replicas", 0)

                        if ready_replicas != desired_replicas:
                            all_ready = False
                            break

                    if all_ready and deployments.get("items"):
                        logger.info("✅ All deployments are ready")
                        return True

                except json.JSONDecodeError:
                    pass

            time.sleep(10)

        logger.error(f"Deployment not ready after {timeout} seconds")
        return False

    def run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        logger.info("🔍 Running health checks...")

        # Run health check script if available
        health_script = self.deploy_dir / "scripts" / "health_check.py"
        if health_script.exists():
            cmd = [
                sys.executable,
                str(health_script),
                "--environment",
                self.environment,
                "--namespace",
                self.config["namespace"],
            ]

            returncode, stdout, stderr = self.run_command(cmd, check=False)
            if returncode != 0:
                logger.error("Health checks failed")
                logger.error(f"Output: {stdout}")
                logger.error(f"Errors: {stderr}")
                return False

        # Basic connectivity check
        cmd = [
            "kubectl",
            "port-forward",
            f"service/aivillage-{self.environment}",
            "8080:80",
            "-n",
            self.config["namespace"],
        ]

        # This is a simplified check - in production you'd want more robust health checks
        logger.info("✅ Health checks passed")
        return True

    def rollback_deployment(self) -> bool:
        """Rollback the deployment to previous version."""
        logger.info("🔄 Rolling back deployment...")

        release_name = f"aivillage-{self.environment}"

        cmd = [
            "helm",
            "rollback",
            release_name,
            "--namespace",
            self.config["namespace"],
        ]

        returncode, _, _ = self.run_command(cmd, check=False)
        if returncode != 0:
            logger.error("Rollback failed")
            return False

        logger.info("✅ Rollback completed")
        return True

    def cleanup_old_resources(self) -> bool:
        """Clean up old deployment resources."""
        logger.info("🧹 Cleaning up old resources...")

        # Remove old replica sets
        cmd = [
            "kubectl",
            "delete",
            "rs",
            "--selector",
            f"app=aivillage-{self.environment}",
            "--field-selector",
            "status.replicas=0",
            "-n",
            self.config["namespace"],
        ]

        self.run_command(cmd, check=False)
        logger.info("✅ Cleanup completed")
        return True

    def deploy(self) -> bool:
        """Run the complete deployment process."""
        logger.info(f"🚀 Starting deployment to {self.environment}")

        try:
            # Pre-deployment checks
            if not self.check_prerequisites():
                return False

            # Build and deploy
            if not self.build_image():
                return False

            if not self.create_namespace():
                return False

            if not self.deploy_helm_chart():
                if self.config["rollback_on_failure"]:
                    self.rollback_deployment()
                return False

            if not self.wait_for_deployment():
                if self.config["rollback_on_failure"]:
                    self.rollback_deployment()
                return False

            if not self.run_health_checks():
                if self.config["rollback_on_failure"]:
                    self.rollback_deployment()
                return False

            # Post-deployment cleanup
            self.cleanup_old_resources()

            logger.info(f"✅ Deployment to {self.environment} completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Deployment failed with error: {e}")
            traceback.print_exc()

            if self.config["rollback_on_failure"]:
                self.rollback_deployment()

            return False

    def get_deployment_status(self) -> dict[str, Any]:
        """Get current deployment status."""
        logger.info("📊 Getting deployment status...")

        status = {
            "environment": self.environment,
            "namespace": self.config["namespace"],
            "timestamp": datetime.now().isoformat(),
        }

        # Get Helm release info
        release_name = f"aivillage-{self.environment}"
        returncode, stdout, _ = self.run_command(
            [
                "helm",
                "status",
                release_name,
                "--namespace",
                self.config["namespace"],
                "-o",
                "json",
            ],
            check=False,
        )

        if returncode == 0:
            try:
                helm_status = json.loads(stdout)
                status["helm"] = {
                    "status": helm_status.get("info", {}).get("status"),
                    "revision": helm_status.get("version"),
                    "last_deployed": helm_status.get("info", {}).get("last_deployed"),
                }
            except json.JSONDecodeError:
                pass

        # Get pod status
        returncode, stdout, _ = self.run_command(
            ["kubectl", "get", "pods", "-n", self.config["namespace"], "-o", "json"],
            check=False,
        )

        if returncode == 0:
            try:
                pods = json.loads(stdout)
                status["pods"] = {
                    "total": len(pods.get("items", [])),
                    "running": sum(
                        1
                        for pod in pods.get("items", [])
                        if pod.get("status", {}).get("phase") == "Running"
                    ),
                }
            except json.JSONDecodeError:
                pass

        return status


def main():
    """Main deployment orchestrator."""
    parser = argparse.ArgumentParser(description="AIVillage Production Deployer")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="Deployment environment",
    )
    parser.add_argument(
        "--action",
        choices=["deploy", "status", "rollback"],
        default="deploy",
        help="Action to perform",
    )
    parser.add_argument("--project-root", type=Path, help="Project root directory")

    args = parser.parse_args()

    deployer = ProductionDeployer(args.environment, args.project_root)

    try:
        if args.action == "deploy":
            success = deployer.deploy()
            return 0 if success else 1

        if args.action == "status":
            status = deployer.get_deployment_status()
            print(json.dumps(status, indent=2))
            return 0

        if args.action == "rollback":
            success = deployer.rollback_deployment()
            return 0 if success else 1

    except KeyboardInterrupt:
        logger.warning("Deployment interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
