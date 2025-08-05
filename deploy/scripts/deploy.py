#!/usr/bin/env python3
"""Comprehensive deployment orchestration script for AIVillage.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AIVillageDeploymentOrchestrator:
    def __init__(self, environment: str, namespace: str, image_tag: str = "latest"):
        self.environment = environment
        self.namespace = namespace
        self.image_tag = image_tag
        self.helm_chart_path = Path(__file__).parent.parent / "helm" / "aivillage"
        self.k8s_manifests_path = Path(__file__).parent.parent / "k8s"

    def run_command(self, cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a command with error handling."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {e}")
            raise

    def check_prerequisites(self) -> bool:
        """Check that all required tools are available."""
        logger.info("🔍 Checking prerequisites...")

        required_tools = ["kubectl", "helm", "docker"]
        missing_tools = []

        for tool in required_tools:
            try:
                self.run_command([tool, "version"], timeout=30)
                logger.info(f"✅ {tool} is available")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(tool)
                logger.error(f"❌ {tool} is not available")

        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False

        # Check Kubernetes connectivity
        try:
            self.run_command(["kubectl", "cluster-info"], timeout=30)
            logger.info("✅ Kubernetes cluster is accessible")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Kubernetes cluster: {e}")
            return False

        logger.info("✅ All prerequisites satisfied")
        return True

    def create_namespace(self) -> bool:
        """Create the namespace if it doesn't exist."""
        logger.info(f"📦 Creating namespace {self.namespace}...")

        try:
            # Check if namespace exists
            self.run_command(["kubectl", "get", "namespace", self.namespace])
            logger.info(f"✅ Namespace {self.namespace} already exists")
            return True
        except subprocess.CalledProcessError:
            # Namespace doesn't exist, create it
            try:
                manifest_file = self.k8s_manifests_path / "namespace.yaml"
                self.run_command(["kubectl", "apply", "-f", str(manifest_file)])
                logger.info(f"✅ Created namespace {self.namespace}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to create namespace: {e}")
                return False

    def deploy_databases(self) -> bool:
        """Deploy database services."""
        logger.info("🗄️ Deploying database services...")

        database_manifests = ["postgres.yaml", "redis.yaml", "neo4j.yaml", "qdrant.yaml"]

        try:
            for manifest in database_manifests:
                manifest_file = self.k8s_manifests_path / manifest
                if manifest_file.exists():
                    self.run_command(["kubectl", "apply", "-f", str(manifest_file), "-n", self.namespace])
                    logger.info(f"✅ Applied {manifest}")
                else:
                    logger.warning(f"⚠️ Manifest {manifest} not found")

            # Wait for databases to be ready
            logger.info("⏳ Waiting for databases to be ready...")
            self.wait_for_databases()

            logger.info("✅ Database services deployed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to deploy databases: {e}")
            return False

    def wait_for_databases(self, timeout: int = 600) -> bool:
        """Wait for all databases to be ready."""
        databases = [
            ("statefulset", "aivillage-postgres"),
            ("statefulset", "aivillage-redis"),
            ("statefulset", "aivillage-neo4j"),
            ("statefulset", "aivillage-qdrant"),
        ]

        start_time = time.time()

        for resource_type, resource_name in databases:
            logger.info(f"⏳ Waiting for {resource_name} to be ready...")

            while time.time() - start_time < timeout:
                try:
                    result = self.run_command(
                        [
                            "kubectl",
                            "get",
                            resource_type,
                            resource_name,
                            "-n",
                            self.namespace,
                            "-o",
                            "jsonpath={.status.readyReplicas}",
                        ]
                    )

                    if result.stdout.strip() and int(result.stdout.strip()) > 0:
                        logger.info(f"✅ {resource_name} is ready")
                        break

                except Exception:
                    pass

                time.sleep(10)
            else:
                logger.error(f"❌ {resource_name} did not become ready within timeout")
                return False

        return True

    def deploy_with_helm(self) -> bool:
        """Deploy the application using Helm."""
        logger.info("🚀 Deploying application with Helm...")

        try:
            # Build Helm command
            helm_cmd = [
                "helm",
                "upgrade",
                "--install",
                f"aivillage-{self.environment.lower()}",
                str(self.helm_chart_path),
                "--namespace",
                self.namespace,
                "--create-namespace",
                "--set",
                f"image.tag={self.image_tag}",
                "--set",
                f"environment={self.environment}",
                "--values",
                str(self.helm_chart_path / f"values-{self.environment.lower()}.yaml"),
                "--wait",
                "--timeout=15m",
            ]

            # Add environment-specific configurations
            if self.environment.lower() == "production":
                helm_cmd.extend(
                    [
                        "--set",
                        "deployment.replicaCount=5",
                        "--set",
                        "resources.requests.cpu=20",
                        "--set",
                        "resources.requests.memory=40Gi",
                    ]
                )
            elif self.environment.lower() == "staging":
                helm_cmd.extend(
                    [
                        "--set",
                        "deployment.replicaCount=2",
                        "--set",
                        "resources.requests.cpu=5",
                        "--set",
                        "resources.requests.memory=10Gi",
                    ]
                )

            self.run_command(helm_cmd, timeout=900)  # 15 minute timeout
            logger.info("✅ Application deployed successfully with Helm")
            return True

        except Exception as e:
            logger.error(f"❌ Helm deployment failed: {e}")
            return False

    def verify_deployment(self) -> bool:
        """Verify the deployment is working correctly."""
        logger.info("🔍 Verifying deployment...")

        try:
            # Check pod status
            result = self.run_command(
                ["kubectl", "get", "pods", "-n", self.namespace, "-o", "jsonpath={.items[*].status.phase}"]
            )

            phases = result.stdout.strip().split()
            running_pods = phases.count("Running")
            total_pods = len(phases)

            if running_pods == total_pods and total_pods > 0:
                logger.info(f"✅ All pods are running ({running_pods}/{total_pods})")
            else:
                logger.error(f"❌ Not all pods are running ({running_pods}/{total_pods})")
                return False

            # Run basic health checks
            health_script = Path(__file__).parent / "health_check.py"
            if health_script.exists():
                result = subprocess.run(
                    ["python", str(health_script), "--environment", self.environment.lower(), "--quiet"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    logger.info("✅ Health checks passed")
                else:
                    logger.error("❌ Health checks failed")
                    return False

            logger.info("✅ Deployment verification completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Deployment verification failed: {e}")
            return False

    def cleanup_old_resources(self) -> bool:
        """Clean up old resources if needed."""
        logger.info("🧹 Cleaning up old resources...")

        try:
            # Remove old ConfigMaps and Secrets (keep current ones)
            cutoff_time = int(time.time()) - 7 * 24 * 3600  # 7 days ago

            # This is a placeholder - implement actual cleanup logic based on your needs
            logger.info("✅ Cleanup completed")
            return True

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False

    def rollback_deployment(self, revision: int | None = None) -> bool:
        """Rollback the deployment to a previous revision."""
        logger.info("🔄 Rolling back deployment...")

        try:
            helm_cmd = ["helm", "rollback", f"aivillage-{self.environment.lower()}", "--namespace", self.namespace]

            if revision:
                helm_cmd.append(str(revision))

            self.run_command(helm_cmd)
            logger.info("✅ Rollback completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

    async def full_deployment(self) -> bool:
        """Run the complete deployment process."""
        logger.info(f"🚀 Starting full deployment for {self.environment} environment")

        deployment_steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Create Namespace", self.create_namespace),
            ("Deploy Databases", self.deploy_databases),
            ("Deploy Application", self.deploy_with_helm),
            ("Verify Deployment", self.verify_deployment),
            ("Cleanup", self.cleanup_old_resources),
        ]

        for step_name, step_func in deployment_steps:
            logger.info(f"📋 Step: {step_name}")

            try:
                if not step_func():
                    logger.error(f"❌ Step '{step_name}' failed")
                    return False
                logger.info(f"✅ Step '{step_name}' completed")
            except Exception as e:
                logger.error(f"❌ Step '{step_name}' failed with exception: {e}")
                return False

        logger.info("🎉 Full deployment completed successfully!")
        return True

    def get_deployment_status(self) -> dict:
        """Get current deployment status."""
        try:
            # Get Helm release info
            result = self.run_command(
                ["helm", "status", f"aivillage-{self.environment.lower()}", "--namespace", self.namespace, "-o", "json"]
            )

            helm_status = json.loads(result.stdout)

            # Get pod status
            result = self.run_command(["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"])

            pods_data = json.loads(result.stdout)

            pod_status = {}
            for pod in pods_data["items"]:
                pod_name = pod["metadata"]["name"]
                pod_status[pod_name] = {
                    "phase": pod["status"]["phase"],
                    "ready": all(cs["ready"] for cs in pod["status"].get("containerStatuses", [])),
                    "restarts": sum(cs.get("restartCount", 0) for cs in pod["status"].get("containerStatuses", [])),
                }

            return {
                "environment": self.environment,
                "namespace": self.namespace,
                "helm_status": helm_status,
                "pod_status": pod_status,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="AIVillage Deployment Orchestrator")
    parser.add_argument(
        "--environment", required=True, choices=["staging", "production"], help="Deployment environment"
    )
    parser.add_argument("--image-tag", default="latest", help="Docker image tag to deploy")
    parser.add_argument(
        "--action", default="deploy", choices=["deploy", "rollback", "status"], help="Action to perform"
    )
    parser.add_argument("--rollback-revision", type=int, help="Revision to rollback to")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")

    args = parser.parse_args()

    namespace = f"aivillage-{args.environment}"
    orchestrator = AIVillageDeploymentOrchestrator(args.environment, namespace, args.image_tag)

    if args.dry_run:
        logger.info("🏃 Dry run mode - showing what would be done")
        print(f"Environment: {args.environment}")
        print(f"Namespace: {namespace}")
        print(f"Image tag: {args.image_tag}")
        print(f"Action: {args.action}")
        return 0

    try:
        if args.action == "deploy":
            success = asyncio.run(orchestrator.full_deployment())
        elif args.action == "rollback":
            success = orchestrator.rollback_deployment(args.rollback_revision)
        elif args.action == "status":
            status = orchestrator.get_deployment_status()
            print(json.dumps(status, indent=2))
            success = "error" not in status
        else:
            logger.error(f"Unknown action: {args.action}")
            success = False

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
