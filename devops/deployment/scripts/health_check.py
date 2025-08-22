#!/usr/bin/env python3
"""Comprehensive health check script for AIVillage deployment."""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self, environment: str) -> None:
        self.environment = environment
        self.namespace = f"aivillage-{environment}"
        self.results = {}
        self.overall_health = True

    async def check_kubernetes_resources(self) -> dict:
        """Check the health of Kubernetes resources."""
        logger.info("Checking Kubernetes resources...")

        try:
            # Get all pods
            cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if result.returncode != 0:
                return {
                    "status": "FAIL",
                    "error": f"Failed to get pods: {result.stderr}",
                }

            pods_data = json.loads(result.stdout)
            pod_health = {}

            for pod in pods_data["items"]:
                pod_name = pod["metadata"]["name"]
                phase = pod["status"]["phase"]
                ready = False

                if "containerStatuses" in pod["status"]:
                    ready = all(cs["ready"] for cs in pod["status"]["containerStatuses"])

                pod_health[pod_name] = {
                    "phase": phase,
                    "ready": ready,
                    "restarts": sum(cs.get("restartCount", 0) for cs in pod["status"].get("containerStatuses", [])),
                }

                if phase != "Running" or not ready:
                    self.overall_health = False

            # Get services
            cmd = ["kubectl", "get", "services", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            services_data = json.loads(result.stdout) if result.returncode == 0 else {"items": []}
            service_health = {svc["metadata"]["name"]: {"type": svc["spec"]["type"]} for svc in services_data["items"]}

            return {
                "status": (
                    "PASS" if all(p["phase"] == "Running" and p["ready"] for p in pod_health.values()) else "FAIL"
                ),
                "pods": pod_health,
                "services": service_health,
            }

        except Exception as e:
            self.overall_health = False
            return {"status": "FAIL", "error": str(e)}

    async def check_database_health(self) -> dict:
        """Check the health of all databases."""
        logger.info("Checking database health...")

        db_results = {}

        # PostgreSQL
        try:
            cmd = [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                "statefulset/aivillage-postgres",
                "--",
                "pg_isready",
                "-h",
                "localhost",
                "-p",
                "5432",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            db_results["postgres"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "details": result.stdout if result.returncode == 0 else result.stderr,
            }
        except Exception as e:
            db_results["postgres"] = {"status": "FAIL", "error": str(e)}
            self.overall_health = False

        # Redis
        try:
            cmd = [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                "statefulset/aivillage-redis",
                "--",
                "redis-cli",
                "ping",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            db_results["redis"] = {
                "status": ("PASS" if result.returncode == 0 and "PONG" in result.stdout else "FAIL"),
                "details": result.stdout if result.returncode == 0 else result.stderr,
            }
        except Exception as e:
            db_results["redis"] = {"status": "FAIL", "error": str(e)}
            self.overall_health = False

        # Neo4j
        try:
            cmd = [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                "statefulset/aivillage-neo4j",
                "--",
                "cypher-shell",
                "-u",
                "neo4j",
                "-p",
                "test_password",
                "RETURN 1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            db_results["neo4j"] = {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "details": result.stdout if result.returncode == 0 else result.stderr,
            }
        except Exception as e:
            db_results["neo4j"] = {"status": "FAIL", "error": str(e)}
            self.overall_health = False

        # Qdrant
        try:
            # Port forward to check Qdrant health
            port_forward_cmd = [
                "kubectl",
                "port-forward",
                "-n",
                self.namespace,
                "statefulset/aivillage-qdrant",
                "6333:6333",
            ]
            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            await asyncio.sleep(2)  # Wait for port forward

            async with (
                aiohttp.ClientSession() as session,
                session.get("http://localhost:6333/", timeout=aiohttp.ClientTimeout(total=10)) as response,
            ):
                if response.status == 200:
                    db_results["qdrant"] = {
                        "status": "PASS",
                        "response_code": response.status,
                    }
                else:
                    db_results["qdrant"] = {
                        "status": "FAIL",
                        "response_code": response.status,
                    }
                    self.overall_health = False

            proc.terminate()

        except Exception as e:
            db_results["qdrant"] = {"status": "FAIL", "error": str(e)}
            self.overall_health = False
            if "proc" in locals():
                proc.terminate()

        return db_results

    async def check_service_endpoints(self) -> dict:
        """Check all service endpoints."""
        logger.info("Checking service endpoints...")

        endpoints = [
            ("gateway", "aivillage-gateway", 8000, "/healthz"),
            ("twin", "aivillage-twin", 8001, "/healthz"),
            ("credits-api", "aivillage-credits-api", 8002, "/health"),
            ("hyperag-mcp", "aivillage-hyperag-mcp", 8765, "/health"),
        ]

        endpoint_results = {}

        for service_name, k8s_service, port, path in endpoints:
            try:
                # Port forward to service
                port_forward_cmd = [
                    "kubectl",
                    "port-forward",
                    "-n",
                    self.namespace,
                    f"service/{k8s_service}",
                    f"{port}:{port}",
                ]
                proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                await asyncio.sleep(2)  # Wait for port forward

                async with aiohttp.ClientSession() as session:
                    url = f"http://localhost:{port}{path}"
                    start_time = time.time()

                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response_time = (time.time() - start_time) * 1000  # ms

                        if response.status == 200:
                            endpoint_results[service_name] = {
                                "status": "PASS",
                                "response_code": response.status,
                                "response_time_ms": round(response_time, 2),
                            }
                        else:
                            endpoint_results[service_name] = {
                                "status": "FAIL",
                                "response_code": response.status,
                                "response_time_ms": round(response_time, 2),
                            }
                            self.overall_health = False

                proc.terminate()

            except Exception as e:
                endpoint_results[service_name] = {"status": "FAIL", "error": str(e)}
                self.overall_health = False
                if "proc" in locals():
                    proc.terminate()

        return endpoint_results

    async def check_resource_usage(self) -> dict:
        """Check resource usage of the cluster."""
        logger.info("Checking resource usage...")

        try:
            # Get node metrics if metrics-server is available
            cmd = ["kubectl", "top", "nodes"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            node_metrics = {}
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        node_name = parts[0]
                        cpu_usage = parts[1]
                        memory_usage = parts[3]
                        node_metrics[node_name] = {
                            "cpu_usage": cpu_usage,
                            "memory_usage": memory_usage,
                        }

            # Get pod metrics
            cmd = ["kubectl", "top", "pods", "-n", self.namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            pod_metrics = {}
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        pod_name = parts[0]
                        cpu_usage = parts[1]
                        memory_usage = parts[2]
                        pod_metrics[pod_name] = {
                            "cpu_usage": cpu_usage,
                            "memory_usage": memory_usage,
                        }

            return {
                "status": "PASS",
                "node_metrics": node_metrics,
                "pod_metrics": pod_metrics,
            }

        except Exception as e:
            return {"status": "FAIL", "error": str(e)}

    async def run_comprehensive_health_check(self) -> dict:
        """Run all health checks."""
        logger.info(f"Starting comprehensive health check for {self.environment} environment...")

        start_time = time.time()

        # Run all checks concurrently
        k8s_check = asyncio.create_task(self.check_kubernetes_resources())
        db_check = asyncio.create_task(self.check_database_health())
        endpoint_check = asyncio.create_task(self.check_service_endpoints())
        resource_check = asyncio.create_task(self.check_resource_usage())

        # Wait for all checks to complete
        k8s_results = await k8s_check
        db_results = await db_check
        endpoint_results = await endpoint_check
        resource_results = await resource_check

        execution_time = time.time() - start_time

        self.results = {
            "environment": self.environment,
            "namespace": self.namespace,
            "timestamp": time.time(),
            "execution_time_seconds": round(execution_time, 2),
            "overall_health": self.overall_health,
            "kubernetes": k8s_results,
            "databases": db_results,
            "service_endpoints": endpoint_results,
            "resource_usage": resource_results,
        }

        return self.results

    def print_health_summary(self) -> None:
        """Print a human-readable health summary."""
        print(f"\n{'=' * 60}")
        print(f"AIVillage Health Check Summary - {self.environment.upper()}")
        print(f"{'=' * 60}")

        overall_status = "🟢 HEALTHY" if self.overall_health else "🔴 UNHEALTHY"
        print(f"Overall Status: {overall_status}")
        print(f"Check Duration: {self.results['execution_time_seconds']}s")
        print()

        # Kubernetes resources
        k8s = self.results.get("kubernetes", {})
        print("📦 Kubernetes Resources:")
        if "pods" in k8s:
            for pod_name, pod_info in k8s["pods"].items():
                status_icon = "🟢" if pod_info["phase"] == "Running" and pod_info["ready"] else "🔴"
                print(f"  {status_icon} {pod_name}: {pod_info['phase']} (Restarts: {pod_info['restarts']})")
        print()

        # Databases
        db = self.results.get("databases", {})
        print("💾 Databases:")
        for db_name, db_info in db.items():
            status_icon = "🟢" if db_info["status"] == "PASS" else "🔴"
            print(f"  {status_icon} {db_name}: {db_info['status']}")
        print()

        # Service endpoints
        endpoints = self.results.get("service_endpoints", {})
        print("🌐 Service Endpoints:")
        for service_name, service_info in endpoints.items():
            status_icon = "🟢" if service_info["status"] == "PASS" else "🔴"
            response_time = service_info.get("response_time_ms", "N/A")
            print(f"  {status_icon} {service_name}: {service_info['status']} ({response_time}ms)")
        print()

        print(f"{'=' * 60}")

    def save_results(self, output_file: str) -> None:
        """Save results to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Health check results saved to {output_file}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive health check for AIVillage deployment")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="Environment to check",
    )
    parser.add_argument("--output", default="health_check_results.json", help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    health_checker = HealthChecker(args.environment)
    await health_checker.run_comprehensive_health_check()

    if not args.quiet:
        health_checker.print_health_summary()

    health_checker.save_results(args.output)

    # Exit with appropriate code
    sys.exit(0 if health_checker.overall_health else 1)


if __name__ == "__main__":
    asyncio.run(main())
