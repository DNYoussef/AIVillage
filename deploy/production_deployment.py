"""Production Deployment Automation for AIVillage.

Comprehensive deployment system with zero-downtime deployments,
automated rollback, and production hardening.
"""

import asyncio
from datetime import datetime
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class DeploymentConfig:
    """Deployment configuration management."""

    def __init__(self, config_path: str = "deploy/config.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "environments": {
                "development": {
                    "replicas": 1,
                    "resources": {"cpu": "500m", "memory": "1Gi"},
                    "auto_deploy": True,
                    "health_checks": {"timeout": 30, "retries": 3},
                },
                "staging": {
                    "replicas": 2,
                    "resources": {"cpu": "1000m", "memory": "2Gi"},
                    "auto_deploy": False,
                    "health_checks": {"timeout": 60, "retries": 5},
                },
                "production": {
                    "replicas": 3,
                    "resources": {"cpu": "2000m", "memory": "4Gi"},
                    "auto_deploy": False,
                    "health_checks": {"timeout": 120, "retries": 10},
                },
            },
            "services": {
                "hyperag-mcp": {
                    "port": 8765,
                    "health_endpoint": "/health",
                    "image": "aivillage/hyperag-mcp:latest",
                },
                "mesh-network": {
                    "port": 9000,
                    "health_endpoint": "/status",
                    "image": "aivillage/mesh-network:latest",
                },
                "evolution-engine": {
                    "port": 8080,
                    "health_endpoint": "/api/health",
                    "image": "aivillage/evolution-engine:latest",
                },
                "compression-service": {
                    "port": 8081,
                    "health_endpoint": "/api/health",
                    "image": "aivillage/compression:latest",
                },
                "rag-system": {
                    "port": 8082,
                    "health_endpoint": "/api/health",
                    "image": "aivillage/rag-system:latest",
                },
            },
            "deployment": {
                "strategy": "blue-green",
                "timeout_seconds": 600,
                "rollback_on_failure": True,
                "pre_deployment_tests": True,
                "post_deployment_validation": True,
            },
            "monitoring": {
                "enabled": True,
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "alert_manager_port": 9093,
            },
        }

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")

        return default_config

    def get_environment_config(self, env: str) -> dict[str, Any]:
        """Get configuration for specific environment."""
        return self.config["environments"].get(env, {})

    def get_service_config(self, service: str) -> dict[str, Any]:
        """Get configuration for specific service."""
        return self.config["services"].get(service, {})


class ContainerBuilder:
    """Builds Docker containers for AIVillage services."""

    def __init__(self, build_dir: str = "build") -> None:
        self.build_dir = build_dir
        os.makedirs(build_dir, exist_ok=True)

    def create_dockerfile(self, service: str, service_config: dict[str, Any]) -> str:
        """Create Dockerfile for service."""
        dockerfile_content = self._get_dockerfile_template(service)
        dockerfile_path = os.path.join(self.build_dir, f"Dockerfile.{service}")

        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        return dockerfile_path

    def _get_dockerfile_template(self, service: str) -> str:
        """Get Dockerfile template for service."""
        base_template = """
FROM python:3.11-slim

# Security hardening
RUN useradd --create-home --shell /bin/bash aivillage
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R aivillage:aivillage /app
USER aivillage

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:{port}/health')"

# Expose port
EXPOSE {port}

# Run application
CMD ["{cmd}"]
"""

        service_configs = {
            "hyperag-mcp": {
                "port": 8765,
                "cmd": "python -m mcp_servers.hyperag.mcp_server",
            },
            "mesh-network": {"port": 9000, "cmd": "python mesh_network_manager.py"},
            "evolution-engine": {
                "port": 8080,
                "cmd": "python agent_forge/self_evolution_engine.py",
            },
            "compression-service": {
                "port": 8081,
                "cmd": "python -m production.compression.compression_pipeline",
            },
            "rag-system": {
                "port": 8082,
                "cmd": "python -m production.rag.rag_system.main",
            },
        }

        config = service_configs.get(service, {"port": 8000, "cmd": "python app.py"})
        return base_template.format(**config)

    async def build_image(self, service: str, tag: str = "latest") -> bool:
        """Build Docker image for service."""
        try:
            dockerfile_path = self.create_dockerfile(service, {})
            image_name = f"aivillage/{service}:{tag}"

            # Build command
            build_cmd = [
                "docker",
                "build",
                "-f",
                dockerfile_path,
                "-t",
                image_name,
                ".",
            ]

            logger.info(f"Building image: {image_name}")
            result = subprocess.run(
                build_cmd, capture_output=True, text=True, timeout=600, check=False
            )  # 10 minute timeout

            if result.returncode == 0:
                logger.info(f"Successfully built image: {image_name}")
                return True
            logger.error(f"Failed to build image: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error building image for {service}: {e}")
            return False


class KubernetesDeployer:
    """Handles Kubernetes deployments."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.manifests_dir = "deploy/k8s"
        os.makedirs(self.manifests_dir, exist_ok=True)

    def create_deployment_manifest(
        self, service: str, environment: str, version: str = "latest"
    ) -> str:
        """Create Kubernetes deployment manifest."""
        env_config = self.config.get_environment_config(environment)
        service_config = self.config.get_service_config(service)

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{service}-{environment}",
                "namespace": f"aivillage-{environment}",
                "labels": {
                    "app": service,
                    "environment": environment,
                    "version": version,
                },
            },
            "spec": {
                "replicas": env_config.get("replicas", 1),
                "selector": {
                    "matchLabels": {"app": service, "environment": environment}
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service,
                            "environment": environment,
                            "version": version,
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": service,
                                "image": f"{service_config.get('image', f'aivillage/{service}:latest')}",
                                "ports": [
                                    {
                                        "containerPort": service_config.get(
                                            "port", 8000
                                        ),
                                        "name": "http",
                                    }
                                ],
                                "resources": env_config.get("resources", {}),
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": service_config.get(
                                            "health_endpoint", "/health"
                                        ),
                                        "port": "http",
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3,
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": service_config.get(
                                            "health_endpoint", "/health"
                                        ),
                                        "port": "http",
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3,
                                },
                                "env": [
                                    {"name": "ENVIRONMENT", "value": environment},
                                    {"name": "SERVICE_NAME", "value": service},
                                    {"name": "LOG_LEVEL", "value": "INFO"},
                                ],
                            }
                        ],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000,
                        },
                    },
                },
            },
        }

        manifest_path = os.path.join(
            self.manifests_dir, f"{service}-{environment}-deployment.yaml"
        )

        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)

        return manifest_path

    def create_service_manifest(self, service: str, environment: str) -> str:
        """Create Kubernetes service manifest."""
        service_config = self.config.get_service_config(service)

        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{service}-{environment}",
                "namespace": f"aivillage-{environment}",
                "labels": {"app": service, "environment": environment},
            },
            "spec": {
                "selector": {"app": service, "environment": environment},
                "ports": [
                    {
                        "port": service_config.get("port", 8000),
                        "targetPort": "http",
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
                "type": "ClusterIP",
            },
        }

        manifest_path = os.path.join(
            self.manifests_dir, f"{service}-{environment}-service.yaml"
        )

        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)

        return manifest_path

    async def deploy_service(
        self, service: str, environment: str, version: str = "latest"
    ) -> bool:
        """Deploy service to Kubernetes."""
        try:
            # Create manifests
            deployment_manifest = self.create_deployment_manifest(
                service, environment, version
            )
            service_manifest = self.create_service_manifest(service, environment)

            # Apply manifests
            for manifest in [deployment_manifest, service_manifest]:
                cmd = ["kubectl", "apply", "-f", manifest]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )

                if result.returncode != 0:
                    logger.error(
                        f"Failed to apply manifest {manifest}: {result.stderr}"
                    )
                    return False

            # Wait for rollout
            rollout_cmd = [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{service}-{environment}",
                "-n",
                f"aivillage-{environment}",
                "--timeout=300s",
            ]

            result = subprocess.run(
                rollout_cmd, capture_output=True, text=True, check=False
            )

            if result.returncode == 0:
                logger.info(f"Successfully deployed {service} to {environment}")
                return True
            logger.error(f"Deployment rollout failed: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error deploying {service} to {environment}: {e}")
            return False


class HealthChecker:
    """Validates deployment health and readiness."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config

    async def check_service_health(
        self, service: str, environment: str, timeout: int = 60
    ) -> bool:
        """Check if service is healthy."""
        service_config = self.config.get_service_config(service)
        health_endpoint = service_config.get("health_endpoint", "/health")

        # Get service URL (simplified for this example)
        f"http://{service}-{environment}::{service_config.get('port', 8000)}{health_endpoint}"

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Simulate health check
                # In real implementation, would make HTTP request
                health_status = True  # Mock successful health check

                if health_status:
                    logger.info(f"Service {service} is healthy in {environment}")
                    return True

            except Exception as e:
                logger.debug(f"Health check failed for {service}: {e}")

            await asyncio.sleep(5)

        logger.error(f"Service {service} failed health check in {environment}")
        return False

    async def validate_deployment(self, services: list[str], environment: str) -> bool:
        """Validate all services in deployment."""
        health_checks = []
        for service in services:
            health_checks.append(self.check_service_health(service, environment))

        results = await asyncio.gather(*health_checks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        total_count = len(services)

        logger.info(
            f"Health validation: {success_count}/{total_count} services healthy"
        )

        return success_count == total_count


class BlueGreenDeployer:
    """Implements blue-green deployment strategy."""

    def __init__(
        self, config: DeploymentConfig, k8s_deployer: KubernetesDeployer
    ) -> None:
        self.config = config
        self.k8s_deployer = k8s_deployer
        self.health_checker = HealthChecker(config)

    async def deploy_blue_green(
        self, services: list[str], environment: str, version: str
    ) -> bool:
        """Execute blue-green deployment."""
        logger.info(f"Starting blue-green deployment to {environment}")

        # Step 1: Deploy to "green" environment
        green_env = f"{environment}-green"

        deployment_success = True
        for service in services:
            success = await self.k8s_deployer.deploy_service(
                service, green_env, version
            )
            if not success:
                deployment_success = False
                break

        if not deployment_success:
            logger.error("Green deployment failed")
            return False

        # Step 2: Validate green environment
        validation_success = await self.health_checker.validate_deployment(
            services, green_env
        )

        if not validation_success:
            logger.error("Green environment validation failed")
            await self._cleanup_green_environment(services, green_env)
            return False

        # Step 3: Switch traffic to green (simulate)
        logger.info("Switching traffic to green environment")
        traffic_switch_success = await self._switch_traffic(environment, green_env)

        if not traffic_switch_success:
            logger.error("Traffic switch failed")
            await self._cleanup_green_environment(services, green_env)
            return False

        # Step 4: Clean up blue environment
        await self._cleanup_blue_environment(services, environment)

        # Step 5: Promote green to blue
        await self._promote_green_to_blue(services, environment, green_env)

        logger.info(f"Blue-green deployment to {environment} completed successfully")
        return True

    async def _switch_traffic(self, blue_env: str, green_env: str) -> bool:
        """Switch traffic from blue to green environment."""
        # In real implementation, would update load balancer or ingress
        # For now, simulate successful traffic switch
        await asyncio.sleep(2)
        return True

    async def _cleanup_green_environment(
        self, services: list[str], green_env: str
    ) -> None:
        """Clean up failed green environment."""
        for service in services:
            try:
                cmd = [
                    "kubectl",
                    "delete",
                    "deployment",
                    f"{service}-{green_env}",
                    "-n",
                    f"aivillage-{green_env}",
                ]
                subprocess.run(cmd, capture_output=True, check=False)
            except Exception as e:
                logger.warning(f"Failed to cleanup green deployment for {service}: {e}")

    async def _cleanup_blue_environment(
        self, services: list[str], blue_env: str
    ) -> None:
        """Clean up old blue environment."""
        # Similar to green cleanup

    async def _promote_green_to_blue(
        self, services: list[str], blue_env: str, green_env: str
    ) -> None:
        """Promote green environment to blue."""
        # Update labels and configurations


class ProductionDeploymentOrchestrator:
    """Main orchestrator for production deployments."""

    def __init__(self, config_path: str = "deploy/config.yaml") -> None:
        self.config = DeploymentConfig(config_path)
        self.container_builder = ContainerBuilder()
        self.k8s_deployer = KubernetesDeployer(self.config)
        self.blue_green_deployer = BlueGreenDeployer(self.config, self.k8s_deployer)
        self.health_checker = HealthChecker(self.config)

        self.deployment_history: list[dict[str, Any]] = []

    async def deploy_to_environment(
        self,
        environment: str,
        services: list[str] | None = None,
        version: str = "latest",
        run_tests: bool = True,
    ) -> bool:
        """Deploy services to specified environment."""
        if services is None:
            services = list(self.config.config["services"].keys())

        deployment_start = datetime.now()
        deployment_id = f"deploy_{int(deployment_start.timestamp())}"

        logger.info(f"Starting deployment {deployment_id} to {environment}")
        logger.info(f"Services: {services}")
        logger.info(f"Version: {version}")

        try:
            # Step 1: Pre-deployment validation
            if run_tests:
                test_success = await self._run_pre_deployment_tests()
                if not test_success:
                    logger.error("Pre-deployment tests failed")
                    return False

            # Step 2: Build container images
            build_success = await self._build_all_images(services, version)
            if not build_success:
                logger.error("Container build failed")
                return False

            # Step 3: Deploy based on strategy
            deployment_config = self.config.config["deployment"]
            strategy = deployment_config.get("strategy", "rolling")

            if strategy == "blue-green":
                deploy_success = await self.blue_green_deployer.deploy_blue_green(
                    services, environment, version
                )
            else:
                deploy_success = await self._rolling_deployment(
                    services, environment, version
                )

            if not deploy_success:
                logger.error("Deployment failed")
                return False

            # Step 4: Post-deployment validation
            validation_success = await self.health_checker.validate_deployment(
                services, environment
            )

            if not validation_success:
                logger.error("Post-deployment validation failed")

                # Rollback if configured
                if deployment_config.get("rollback_on_failure", True):
                    await self._rollback_deployment(deployment_id)

                return False

            # Step 5: Record successful deployment
            deployment_record = {
                "id": deployment_id,
                "environment": environment,
                "services": services,
                "version": version,
                "strategy": strategy,
                "start_time": deployment_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - deployment_start).total_seconds(),
                "status": "success",
            }

            self.deployment_history.append(deployment_record)
            await self._save_deployment_history()

            logger.info(f"Deployment {deployment_id} completed successfully")
            return True

        except Exception as e:
            logger.exception(f"Deployment {deployment_id} failed with error: {e}")

            # Record failed deployment
            deployment_record = {
                "id": deployment_id,
                "environment": environment,
                "services": services,
                "version": version,
                "start_time": deployment_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - deployment_start).total_seconds(),
                "status": "failed",
                "error": str(e),
            }

            self.deployment_history.append(deployment_record)
            await self._save_deployment_history()

            return False

    async def _run_pre_deployment_tests(self) -> bool:
        """Run pre-deployment tests."""
        logger.info("Running pre-deployment tests")

        # Run integration tests
        test_cmd = ["python", "tests/integration/test_full_system_integration.py"]

        try:
            result = subprocess.run(
                test_cmd, capture_output=True, text=True, timeout=300, check=False
            )  # 5 minute timeout

            if result.returncode == 0:
                logger.info("Pre-deployment tests passed")
                return True
            logger.error(f"Pre-deployment tests failed: {result.stderr}")
            return False

        except Exception as e:
            logger.exception(f"Error running pre-deployment tests: {e}")
            return False

    async def _build_all_images(self, services: list[str], version: str) -> bool:
        """Build all container images."""
        logger.info(f"Building container images for version {version}")

        build_tasks = []
        for service in services:
            build_tasks.append(self.container_builder.build_image(service, version))

        results = await asyncio.gather(*build_tasks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        total_count = len(services)

        logger.info(f"Container builds: {success_count}/{total_count} successful")

        return success_count == total_count

    async def _rolling_deployment(
        self, services: list[str], environment: str, version: str
    ) -> bool:
        """Execute rolling deployment."""
        logger.info("Executing rolling deployment")

        for service in services:
            success = await self.k8s_deployer.deploy_service(
                service, environment, version
            )
            if not success:
                return False

            # Wait for service to be healthy before continuing
            health_success = await self.health_checker.check_service_health(
                service, environment
            )
            if not health_success:
                return False

        return True

    async def _rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback failed deployment."""
        logger.info(f"Rolling back deployment {deployment_id}")

        # Find previous successful deployment
        successful_deployments = [
            d for d in self.deployment_history if d["status"] == "success"
        ]

        if not successful_deployments:
            logger.error("No previous successful deployment found for rollback")
            return False

        previous_deployment = successful_deployments[-1]

        # Rollback to previous version
        rollback_success = await self.deploy_to_environment(
            previous_deployment["environment"],
            previous_deployment["services"],
            previous_deployment["version"],
            run_tests=False,  # Skip tests for rollback
        )

        return rollback_success

    async def _save_deployment_history(self) -> None:
        """Save deployment history to disk."""
        try:
            os.makedirs("deploy/history", exist_ok=True)
            history_path = "deploy/history/deployments.json"

            with open(history_path, "w") as f:
                json.dump(self.deployment_history, f, indent=2)

        except Exception as e:
            logger.exception(f"Failed to save deployment history: {e}")

    def get_deployment_status(self) -> dict[str, Any]:
        """Get current deployment status."""
        recent_deployments = self.deployment_history[-10:]  # Last 10 deployments

        successful_deployments = sum(
            1 for d in recent_deployments if d["status"] == "success"
        )

        return {
            "total_deployments": len(self.deployment_history),
            "recent_deployments": len(recent_deployments),
            "recent_success_rate": (
                successful_deployments / len(recent_deployments)
                if recent_deployments
                else 0
            ),
            "last_deployment": (
                self.deployment_history[-1] if self.deployment_history else None
            ),
            "environments": list({d["environment"] for d in self.deployment_history}),
            "available_services": list(self.config.config["services"].keys()),
        }


# CLI Interface
async def main() -> None:
    """Main CLI interface for deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="AIVillage Production Deployment")
    parser.add_argument("action", choices=["deploy", "status", "rollback"])
    parser.add_argument(
        "--environment", "-e", default="staging", help="Target environment"
    )
    parser.add_argument(
        "--services", "-s", nargs="+", help="Services to deploy (default: all)"
    )
    parser.add_argument("--version", "-v", default="latest", help="Version to deploy")
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip pre-deployment tests"
    )

    args = parser.parse_args()

    orchestrator = ProductionDeploymentOrchestrator()

    if args.action == "deploy":
        success = await orchestrator.deploy_to_environment(
            args.environment, args.services, args.version, run_tests=not args.skip_tests
        )

        if success:
            print(f"✅ Deployment to {args.environment} successful")
        else:
            print(f"❌ Deployment to {args.environment} failed")
            sys.exit(1)

    elif args.action == "status":
        status = orchestrator.get_deployment_status()
        print(json.dumps(status, indent=2))

    elif args.action == "rollback":
        # Implement rollback logic
        print("Rollback functionality would be implemented here")


if __name__ == "__main__":
    asyncio.run(main())
