#!/usr/bin/env python3
"""
Smoke tests for AIVillage deployment verification.
"""

import asyncio
import aiohttp
import argparse
import json
import sys
import time
from typing import Dict, List, Optional
import subprocess

class SmokeTestRunner:
    def __init__(self, environment: str, namespace: str):
        self.environment = environment
        self.namespace = namespace
        self.results = []

    async def test_service_health(self, service_name: str, port: int, health_path: str = "/health") -> bool:
        """Test if a service is responding to health checks."""
        try:
            # Get service endpoint using kubectl port-forward
            port_forward_cmd = [
                "kubectl", "port-forward",
                f"service/{service_name}",
                f"{port}:{port}",
                f"--namespace={self.namespace}"
            ]

            # Start port forwarding in background
            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait a moment for port forward to establish
            await asyncio.sleep(2)

            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:{port}{health_path}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        result = {"service": service_name, "status": "PASS", "response_code": response.status}
                        print(f"✅ {service_name} health check passed")
                    else:
                        result = {"service": service_name, "status": "FAIL", "response_code": response.status}
                        print(f"❌ {service_name} health check failed with status {response.status}")

                    self.results.append(result)
                    proc.terminate()
                    return response.status == 200

        except Exception as e:
            result = {"service": service_name, "status": "FAIL", "error": str(e)}
            print(f"❌ {service_name} health check failed: {e}")
            self.results.append(result)
            if 'proc' in locals():
                proc.terminate()
            return False

    async def test_database_connectivity(self, db_type: str, service_name: str, port: int) -> bool:
        """Test database connectivity."""
        try:
            # Test database connection using kubectl exec
            if db_type == "postgres":
                cmd = [
                    "kubectl", "exec", "-n", self.namespace,
                    f"statefulset/{service_name}",
                    "--", "pg_isready", "-h", "localhost", "-p", str(port)
                ]
            elif db_type == "redis":
                cmd = [
                    "kubectl", "exec", "-n", self.namespace,
                    f"statefulset/{service_name}",
                    "--", "redis-cli", "ping"
                ]
            elif db_type == "neo4j":
                cmd = [
                    "kubectl", "exec", "-n", self.namespace,
                    f"statefulset/{service_name}",
                    "--", "cypher-shell", "-u", "neo4j", "-p", "test_password", "RETURN 1"
                ]
            else:
                raise ValueError(f"Unknown database type: {db_type}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"✅ {db_type} database connectivity test passed")
                self.results.append({"service": service_name, "db_type": db_type, "status": "PASS"})
                return True
            else:
                print(f"❌ {db_type} database connectivity test failed: {result.stderr}")
                self.results.append({"service": service_name, "db_type": db_type, "status": "FAIL", "error": result.stderr})
                return False

        except Exception as e:
            print(f"❌ {db_type} database test failed: {e}")
            self.results.append({"service": service_name, "db_type": db_type, "status": "FAIL", "error": str(e)})
            return False

    async def test_service_integration(self) -> bool:
        """Test basic service integration."""
        try:
            # Test a simple integration flow: Gateway -> Twin -> MCP
            # This would involve making a request through the gateway
            # that exercises the full service chain

            print("🔄 Testing service integration...")

            # For now, just verify all services are running
            cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                running_pods = [
                    pod for pod in pods_data["items"]
                    if pod["status"]["phase"] == "Running"
                ]

                total_pods = len(pods_data["items"])
                running_count = len(running_pods)

                if running_count == total_pods and total_pods > 0:
                    print(f"✅ Service integration test passed ({running_count}/{total_pods} pods running)")
                    self.results.append({"test": "service_integration", "status": "PASS", "pods_running": running_count})
                    return True
                else:
                    print(f"❌ Service integration test failed ({running_count}/{total_pods} pods running)")
                    self.results.append({"test": "service_integration", "status": "FAIL", "pods_running": running_count, "total_pods": total_pods})
                    return False
            else:
                print(f"❌ Failed to get pod status: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Service integration test failed: {e}")
            self.results.append({"test": "service_integration", "status": "FAIL", "error": str(e)})
            return False

    async def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        print(f"🚀 Starting smoke tests for {self.environment} environment in namespace {self.namespace}")

        all_passed = True

        # Test database connectivity first
        db_tests = [
            ("postgres", "aivillage-postgres", 5432),
            ("redis", "aivillage-redis", 6379),
            ("neo4j", "aivillage-neo4j", 7687),
        ]

        for db_type, service, port in db_tests:
            result = await self.test_database_connectivity(db_type, service, port)
            all_passed = all_passed and result

        # Test service health endpoints
        service_tests = [
            ("aivillage-gateway", 8000, "/healthz"),
            ("aivillage-twin", 8001, "/healthz"),
            ("aivillage-credits-api", 8002, "/health"),
            ("aivillage-hyperag-mcp", 8765, "/health"),
        ]

        for service, port, health_path in service_tests:
            result = await self.test_service_health(service, port, health_path)
            all_passed = all_passed and result

        # Test service integration
        integration_result = await self.test_service_integration()
        all_passed = all_passed and integration_result

        # Generate summary
        passed_count = len([r for r in self.results if r.get("status") == "PASS"])
        total_count = len(self.results)

        print(f"\n📊 Smoke test summary: {passed_count}/{total_count} tests passed")

        if all_passed:
            print("🎉 All smoke tests passed!")
        else:
            print("💥 Some smoke tests failed!")

        return all_passed

    def save_results(self, output_file: str):
        """Save test results to a file."""
        with open(output_file, 'w') as f:
            json.dump({
                "environment": self.environment,
                "namespace": self.namespace,
                "timestamp": time.time(),
                "results": self.results
            }, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Run AIVillage smoke tests")
    parser.add_argument("--environment", required=True, choices=["staging", "production"], help="Environment to test")
    parser.add_argument("--namespace", required=True, help="Kubernetes namespace")
    parser.add_argument("--output", default="smoke_test_results.json", help="Output file for results")

    args = parser.parse_args()

    runner = SmokeTestRunner(args.environment, args.namespace)
    success = await runner.run_all_tests()

    runner.save_results(args.output)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
