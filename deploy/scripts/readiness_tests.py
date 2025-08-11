#!/usr/bin/env python3
"""Production readiness tests for AIVillage deployment."""

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import subprocess
import sys
import time

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ReadinessTest:
    name: str
    description: str
    critical: bool = True
    timeout: int = 60


class ProductionReadinessValidator:
    def __init__(self, environment: str, namespace: str, slot: str) -> None:
        self.environment = environment
        self.namespace = namespace
        self.slot = slot
        self.results = []
        self.all_passed = True

    async def test_service_availability(self, service_name: str, port: int, path: str = "/health") -> bool:
        """Test service availability and response time."""
        test = ReadinessTest(
            name=f"{service_name}_availability",
            description=f"Check {service_name} service availability",
        )

        try:
            # Port forward to service
            port_forward_cmd = [
                "kubectl",
                "port-forward",
                "-n",
                self.namespace,
                f"service/{service_name}",
                f"{port}:{port}",
            ]

            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await asyncio.sleep(3)  # Wait for port forward

            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                # Test multiple requests to ensure stability
                success_count = 0
                total_requests = 5
                response_times = []

                for i in range(total_requests):
                    try:
                        request_start = time.time()
                        async with session.get(
                            f"http://localhost:{port}{path}",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            request_time = (time.time() - request_start) * 1000
                            response_times.append(request_time)

                            if response.status == 200:
                                success_count += 1

                            await asyncio.sleep(0.5)  # Small delay between requests

                    except Exception as e:
                        logger.warning(f"Request {i + 1} failed for {service_name}: {e}")
                        continue

            proc.terminate()

            # Calculate metrics
            success_rate = (success_count / total_requests) * 100
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Define success criteria
            success = success_rate >= 80 and avg_response_time < 5000  # 80% success rate, <5s response time

            result = {
                "test": test.name,
                "status": "PASS" if success else "FAIL",
                "success_rate": success_rate,
                "avg_response_time_ms": round(avg_response_time, 2),
                "total_requests": total_requests,
                "successful_requests": success_count,
            }

            if success:
                logger.info(
                    f"✅ {service_name} availability test passed ({success_rate}% success, {avg_response_time:.2f}ms avg)"
                )
            else:
                logger.error(
                    f"❌ {service_name} availability test failed ({success_rate}% success, {avg_response_time:.2f}ms avg)"
                )
                self.all_passed = False

            self.results.append(result)
            return success

        except Exception as e:
            logger.exception(f"❌ {service_name} availability test failed: {e}")
            self.results.append({"test": test.name, "status": "FAIL", "error": str(e)})
            self.all_passed = False
            if "proc" in locals():
                proc.terminate()
            return False

    async def test_database_performance(self, db_type: str, service_name: str) -> bool:
        """Test database performance and connectivity."""
        test = ReadinessTest(
            name=f"{db_type}_performance",
            description=f"Check {db_type} database performance",
        )

        try:
            start_time = time.time()

            if db_type == "postgres":
                # Test PostgreSQL performance
                cmd = [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    f"statefulset/{service_name}",
                    "--",
                    "psql",
                    "-U",
                    "aivillage_user",
                    "-d",
                    "aivillage_production",
                    "-c",
                    "SELECT COUNT(*) FROM information_schema.tables;",
                ]
            elif db_type == "redis":
                # Test Redis performance
                cmd = [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    f"statefulset/{service_name}",
                    "--",
                    "redis-cli",
                    "--latency-history",
                    "-i",
                    "1",
                ]
            elif db_type == "neo4j":
                # Test Neo4j performance
                cmd = [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    f"statefulset/{service_name}",
                    "--",
                    "cypher-shell",
                    "-u",
                    "neo4j",
                    "-p",
                    "production_password",
                    "MATCH (n) RETURN count(n) LIMIT 1;",
                ]
            else:
                msg = f"Unknown database type: {db_type}"
                raise ValueError(msg)

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            execution_time = (time.time() - start_time) * 1000

            success = result.returncode == 0 and execution_time < 10000  # Less than 10 seconds

            test_result = {
                "test": test.name,
                "status": "PASS" if success else "FAIL",
                "execution_time_ms": round(execution_time, 2),
                "stdout": result.stdout[:200] if result.stdout else "",
                "stderr": result.stderr[:200] if result.stderr else "",
            }

            if success:
                logger.info(f"✅ {db_type} performance test passed ({execution_time:.2f}ms)")
            else:
                logger.error(f"❌ {db_type} performance test failed ({execution_time:.2f}ms)")
                self.all_passed = False

            self.results.append(test_result)
            return success

        except Exception as e:
            logger.exception(f"❌ {db_type} performance test failed: {e}")
            self.results.append({"test": test.name, "status": "FAIL", "error": str(e)})
            self.all_passed = False
            return False

    async def test_resource_limits(self) -> bool:
        """Test that pods are within resource limits."""
        test = ReadinessTest(name="resource_limits", description="Check pod resource usage within limits")

        try:
            # Get pod metrics
            cmd = ["kubectl", "top", "pods", "-n", self.namespace, "--no-headers"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if result.returncode != 0:
                msg = f"Failed to get pod metrics: {result.stderr}"
                raise Exception(msg)

            resource_violations = []

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    pod_name = parts[0]
                    cpu_usage = parts[1]
                    memory_usage = parts[2]

                    # Check if this pod belongs to our deployment slot
                    if self.slot in pod_name:
                        # Parse CPU usage (remove 'm' suffix)
                        cpu_value = int(cpu_usage.replace("m", "")) if "m" in cpu_usage else int(cpu_usage) * 1000

                        # Parse memory usage (convert to Mi)
                        memory_value = int(memory_usage.replace("Mi", ""))

                        # Define resource limits (these should match your deployment specs)
                        cpu_limit = 2000  # 2 CPU = 2000m
                        memory_limit = 4096  # 4Gi = 4096Mi

                        if cpu_value > cpu_limit * 0.8:  # Alert at 80% of limit
                            resource_violations.append(f"{pod_name}: CPU usage {cpu_usage} near limit")

                        if memory_value > memory_limit * 0.8:  # Alert at 80% of limit
                            resource_violations.append(f"{pod_name}: Memory usage {memory_usage} near limit")

            success = len(resource_violations) == 0

            test_result = {
                "test": test.name,
                "status": "PASS" if success else "FAIL",
                "violations": resource_violations,
            }

            if success:
                logger.info("✅ Resource limits test passed")
            else:
                logger.error(f"❌ Resource limits test failed: {resource_violations}")
                self.all_passed = False

            self.results.append(test_result)
            return success

        except Exception as e:
            logger.exception(f"❌ Resource limits test failed: {e}")
            self.results.append({"test": test.name, "status": "FAIL", "error": str(e)})
            self.all_passed = False
            return False

    async def test_load_balancer_configuration(self) -> bool:
        """Test load balancer and ingress configuration."""
        test = ReadinessTest(
            name="load_balancer_config",
            description="Check load balancer and ingress setup",
        )

        try:
            # Check if ingress is properly configured
            cmd = ["kubectl", "get", "ingress", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if result.returncode != 0:
                msg = f"Failed to get ingress: {result.stderr}"
                raise Exception(msg)

            ingress_data = json.loads(result.stdout)
            ingress_items = ingress_data.get("items", [])

            if not ingress_items:
                msg = "No ingress found"
                raise Exception(msg)

            # Check ingress status
            ingress = ingress_items[0]
            ingress_status = ingress.get("status", {})
            load_balancer = ingress_status.get("loadBalancer", {})

            # Check if load balancer has been assigned an IP
            success = bool(load_balancer.get("ingress"))

            test_result = {
                "test": test.name,
                "status": "PASS" if success else "FAIL",
                "ingress_name": ingress["metadata"]["name"],
                "load_balancer_status": load_balancer,
            }

            if success:
                logger.info("✅ Load balancer configuration test passed")
            else:
                logger.error("❌ Load balancer configuration test failed")
                self.all_passed = False

            self.results.append(test_result)
            return success

        except Exception as e:
            logger.exception(f"❌ Load balancer configuration test failed: {e}")
            self.results.append({"test": test.name, "status": "FAIL", "error": str(e)})
            self.all_passed = False
            return False

    async def test_security_configuration(self) -> bool:
        """Test security configurations."""
        test = ReadinessTest(name="security_config", description="Check security configurations")

        try:
            # Check if pods are running as non-root
            cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if result.returncode != 0:
                msg = f"Failed to get pods: {result.stderr}"
                raise Exception(msg)

            pods_data = json.loads(result.stdout)
            security_violations = []

            for pod in pods_data["items"]:
                pod_name = pod["metadata"]["name"]

                # Check if this pod belongs to our deployment slot
                if self.slot in pod_name:
                    spec = pod.get("spec", {})
                    security_context = spec.get("securityContext", {})

                    # Check if running as non-root
                    if not security_context.get("runAsNonRoot", False):
                        security_violations.append(f"{pod_name}: Not configured to run as non-root")

                    # Check containers
                    for container in spec.get("containers", []):
                        container_security = container.get("securityContext", {})

                        if container_security.get("allowPrivilegeEscalation", True):
                            security_violations.append(f"{pod_name}: Privilege escalation allowed")

                        if not container_security.get("readOnlyRootFilesystem", False):
                            security_violations.append(f"{pod_name}: Root filesystem not read-only")

            success = len(security_violations) == 0

            test_result = {
                "test": test.name,
                "status": "PASS" if success else "FAIL",
                "violations": security_violations,
            }

            if success:
                logger.info("✅ Security configuration test passed")
            else:
                logger.error(f"❌ Security configuration test failed: {security_violations}")
                self.all_passed = False

            self.results.append(test_result)
            return success

        except Exception as e:
            logger.exception(f"❌ Security configuration test failed: {e}")
            self.results.append({"test": test.name, "status": "FAIL", "error": str(e)})
            self.all_passed = False
            return False

    async def run_all_readiness_tests(self) -> bool:
        """Run all production readiness tests."""
        logger.info(f"🚀 Starting production readiness tests for {self.environment} environment (slot: {self.slot})")

        start_time = time.time()

        # Service availability tests
        service_tests = [
            ("aivillage-gateway", 8000, "/healthz"),
            ("aivillage-twin", 8001, "/healthz"),
            ("aivillage-credits-api", 8002, "/health"),
            ("aivillage-hyperag-mcp", 8765, "/health"),
        ]

        for service, port, path in service_tests:
            await self.test_service_availability(service, port, path)

        # Database performance tests
        db_tests = [
            ("postgres", "aivillage-postgres"),
            ("redis", "aivillage-redis"),
            ("neo4j", "aivillage-neo4j"),
        ]

        for db_type, service in db_tests:
            await self.test_database_performance(db_type, service)

        # Infrastructure tests
        await self.test_resource_limits()
        await self.test_load_balancer_configuration()
        await self.test_security_configuration()

        execution_time = time.time() - start_time

        # Generate summary
        passed_count = len([r for r in self.results if r.get("status") == "PASS"])
        total_count = len(self.results)

        logger.info(f"\n📊 Production readiness summary: {passed_count}/{total_count} tests passed")
        logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")

        if self.all_passed:
            logger.info("🎉 All production readiness tests passed! Deployment is ready for traffic.")
        else:
            logger.error("💥 Some production readiness tests failed! Review issues before proceeding.")

        return self.all_passed

    def save_results(self, output_file: str) -> None:
        """Save test results to a file."""
        with open(output_file, "w") as f:
            json.dump(
                {
                    "environment": self.environment,
                    "namespace": self.namespace,
                    "slot": self.slot,
                    "timestamp": time.time(),
                    "all_passed": self.all_passed,
                    "results": self.results,
                },
                f,
                indent=2,
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run production readiness tests for AIVillage deployment")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="Environment to test",
    )
    parser.add_argument("--namespace", required=True, help="Kubernetes namespace")
    parser.add_argument("--slot", required=True, choices=["blue", "green"], help="Deployment slot")
    parser.add_argument(
        "--output",
        default="readiness_test_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    validator = ProductionReadinessValidator(args.environment, args.namespace, args.slot)
    success = await validator.run_all_readiness_tests()

    validator.save_results(args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
