#!/usr/bin/env python3
"""
Production verification script for AIVillage deployment.
Validates the production deployment is working correctly after traffic switch.
"""

import asyncio
import aiohttp
import argparse
import json
import sys
import time
import logging
import subprocess
from typing import Dict, List, Optional
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionVerificationSuite:
    def __init__(self, environment: str, slot: str):
        self.environment = environment
        self.slot = slot
        self.namespace = f"aivillage-{environment}"
        self.results = []
        self.verification_passed = True
        
    async def test_end_to_end_workflow(self) -> bool:
        """Test a complete end-to-end workflow through the system."""
        logger.info("🔄 Testing end-to-end workflow...")
        
        try:
            # Get the external IP of the load balancer
            cmd = ["kubectl", "get", "service", "aivillage-active", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"Failed to get service info: {result.stderr}")
            
            service_data = json.loads(result.stdout)
            
            # For testing, we'll use port forwarding
            port_forward_cmd = [
                "kubectl", "port-forward", "-n", self.namespace,
                "service/aivillage-active", "8000:8000"
            ]
            
            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await asyncio.sleep(3)  # Wait for port forward
            
            # Test workflow: Gateway -> Twin -> MCP interaction
            async with aiohttp.ClientSession() as session:
                # 1. Health check
                async with session.get("http://localhost:8000/healthz", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        raise Exception(f"Gateway health check failed: {response.status}")
                
                # 2. Test basic functionality (adjust based on your API)
                test_payload = {
                    "query": "Test production deployment",
                    "type": "health_check"
                }
                
                async with session.post(
                    "http://localhost:8000/api/v1/query",
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info("✅ End-to-end workflow test passed")
                        self.results.append({
                            "test": "end_to_end_workflow",
                            "status": "PASS",
                            "response": response_data
                        })
                        proc.terminate()
                        return True
                    else:
                        raise Exception(f"API request failed: {response.status}")
            
        except Exception as e:
            logger.error(f"❌ End-to-end workflow test failed: {e}")
            self.results.append({
                "test": "end_to_end_workflow",
                "status": "FAIL",
                "error": str(e)
            })
            self.verification_passed = False
            if 'proc' in locals():
                proc.terminate()
            return False
    
    async def test_load_performance(self, concurrent_requests: int = 10, duration_seconds: int = 30) -> bool:
        """Test system performance under load."""
        logger.info(f"🏋️ Testing load performance ({concurrent_requests} concurrent requests for {duration_seconds}s)...")
        
        try:
            # Port forward to the service
            port_forward_cmd = [
                "kubectl", "port-forward", "-n", self.namespace,
                "service/aivillage-active", "8000:8000"
            ]
            
            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await asyncio.sleep(3)
            
            # Track metrics
            response_times = []
            success_count = 0
            error_count = 0
            start_time = time.time()
            
            async def make_request(session, request_id):
                nonlocal success_count, error_count, response_times
                
                request_start = time.time()
                try:
                    async with session.get(
                        "http://localhost:8000/healthz",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        request_time = (time.time() - request_start) * 1000
                        response_times.append(request_time)
                        
                        if response.status == 200:
                            success_count += 1
                        else:
                            error_count += 1
                            
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Request {request_id} failed: {e}")
            
            # Run concurrent requests
            async with aiohttp.ClientSession() as session:
                while time.time() - start_time < duration_seconds:
                    tasks = []
                    for i in range(concurrent_requests):
                        tasks.append(make_request(session, i))
                    
                    await asyncio.gather(*tasks)
                    await asyncio.sleep(0.1)  # Small delay between batches
            
            proc.terminate()
            
            # Calculate metrics
            total_requests = success_count + error_count
            success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0
            
            # Define success criteria
            success = (
                success_rate >= 95 and  # 95% success rate
                avg_response_time < 1000 and  # <1s average response time
                p95_response_time < 2000  # <2s 95th percentile
            )
            
            result = {
                "test": "load_performance",
                "status": "PASS" if success else "FAIL",
                "total_requests": total_requests,
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": round(success_rate, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "p95_response_time_ms": round(p95_response_time, 2)
            }
            
            if success:
                logger.info(f"✅ Load performance test passed ({success_rate}% success, {avg_response_time:.2f}ms avg)")
            else:
                logger.error(f"❌ Load performance test failed ({success_rate}% success, {avg_response_time:.2f}ms avg)")
                self.verification_passed = False
            
            self.results.append(result)
            return success
            
        except Exception as e:
            logger.error(f"❌ Load performance test failed: {e}")
            self.results.append({
                "test": "load_performance",
                "status": "FAIL",
                "error": str(e)
            })
            self.verification_passed = False
            if 'proc' in locals():
                proc.terminate()
            return False
    
    async def test_database_integrity(self) -> bool:
        """Test database integrity and connectivity."""
        logger.info("🗄️ Testing database integrity...")
        
        db_results = {}
        all_db_passed = True
        
        # Test PostgreSQL
        try:
            cmd = [
                "kubectl", "exec", "-n", self.namespace,
                "statefulset/aivillage-postgres",
                "--", "psql", "-U", "aivillage_user", "-d", "aivillage_production",
                "-c", "SELECT version();"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                db_results["postgres"] = {"status": "PASS", "version": result.stdout.strip()}
                logger.info("✅ PostgreSQL integrity test passed")
            else:
                db_results["postgres"] = {"status": "FAIL", "error": result.stderr}
                logger.error(f"❌ PostgreSQL integrity test failed: {result.stderr}")
                all_db_passed = False
                
        except Exception as e:
            db_results["postgres"] = {"status": "FAIL", "error": str(e)}
            all_db_passed = False
        
        # Test Redis
        try:
            cmd = [
                "kubectl", "exec", "-n", self.namespace,
                "statefulset/aivillage-redis",
                "--", "redis-cli", "info", "server"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                db_results["redis"] = {"status": "PASS", "info": result.stdout[:200]}
                logger.info("✅ Redis integrity test passed")
            else:
                db_results["redis"] = {"status": "FAIL", "error": result.stderr}
                logger.error(f"❌ Redis integrity test failed: {result.stderr}")
                all_db_passed = False
                
        except Exception as e:
            db_results["redis"] = {"status": "FAIL", "error": str(e)}
            all_db_passed = False
        
        # Test Neo4j
        try:
            cmd = [
                "kubectl", "exec", "-n", self.namespace,
                "statefulset/aivillage-neo4j",
                "--", "cypher-shell", "-u", "neo4j", "-p", "production_password",
                "CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version;"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                db_results["neo4j"] = {"status": "PASS", "info": result.stdout[:200]}
                logger.info("✅ Neo4j integrity test passed")
            else:
                db_results["neo4j"] = {"status": "FAIL", "error": result.stderr}
                logger.error(f"❌ Neo4j integrity test failed: {result.stderr}")
                all_db_passed = False
                
        except Exception as e:
            db_results["neo4j"] = {"status": "FAIL", "error": str(e)}
            all_db_passed = False
        
        self.results.append({
            "test": "database_integrity",
            "status": "PASS" if all_db_passed else "FAIL",
            "databases": db_results
        })
        
        if not all_db_passed:
            self.verification_passed = False
        
        return all_db_passed
    
    async def test_monitoring_and_alerting(self) -> bool:
        """Test monitoring and alerting systems."""
        logger.info("📊 Testing monitoring and alerting...")
        
        try:
            # Check if Prometheus is collecting metrics
            port_forward_cmd = [
                "kubectl", "port-forward", "-n", self.namespace,
                "service/aivillage-prometheus", "9090:9090"
            ]
            
            proc = subprocess.Popen(port_forward_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            await asyncio.sleep(3)
            
            async with aiohttp.ClientSession() as session:
                # Test Prometheus API
                async with session.get(
                    "http://localhost:9090/api/v1/query?query=up",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        metrics_available = len(data.get("data", {}).get("result", [])) > 0
                        
                        if metrics_available:
                            logger.info("✅ Monitoring system test passed")
                            self.results.append({
                                "test": "monitoring_system",
                                "status": "PASS",
                                "metrics_count": len(data["data"]["result"])
                            })
                            proc.terminate()
                            return True
                        else:
                            raise Exception("No metrics available")
                    else:
                        raise Exception(f"Prometheus API returned {response.status}")
            
        except Exception as e:
            logger.error(f"❌ Monitoring system test failed: {e}")
            self.results.append({
                "test": "monitoring_system",
                "status": "FAIL",
                "error": str(e)
            })
            self.verification_passed = False
            if 'proc' in locals():
                proc.terminate()
            return False
    
    async def test_security_posture(self) -> bool:
        """Test security configurations in production."""
        logger.info("🔒 Testing security posture...")
        
        try:
            # Check network policies
            cmd = ["kubectl", "get", "networkpolicies", "-n", self.namespace, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                policies_data = json.loads(result.stdout)
                policies_count = len(policies_data.get("items", []))
                
                # Check pod security standards
                cmd = ["kubectl", "get", "pods", "-n", self.namespace, "-o", "json"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    pods_data = json.loads(result.stdout)
                    security_violations = []
                    
                    for pod in pods_data["items"]:
                        if self.slot in pod["metadata"]["name"]:
                            # Check security context
                            spec = pod.get("spec", {})
                            security_context = spec.get("securityContext", {})
                            
                            if not security_context.get("runAsNonRoot", False):
                                security_violations.append(f"Pod {pod['metadata']['name']} may run as root")
                            
                            for container in spec.get("containers", []):
                                container_security = container.get("securityContext", {})
                                if container_security.get("allowPrivilegeEscalation", True):
                                    security_violations.append(f"Container in {pod['metadata']['name']} allows privilege escalation")
                    
                    success = len(security_violations) == 0
                    
                    result_data = {
                        "test": "security_posture",
                        "status": "PASS" if success else "FAIL",
                        "network_policies_count": policies_count,
                        "security_violations": security_violations
                    }
                    
                    if success:
                        logger.info("✅ Security posture test passed")
                    else:
                        logger.error(f"❌ Security posture test failed: {security_violations}")
                        self.verification_passed = False
                    
                    self.results.append(result_data)
                    return success
            
            raise Exception("Failed to check security configurations")
            
        except Exception as e:
            logger.error(f"❌ Security posture test failed: {e}")
            self.results.append({
                "test": "security_posture",
                "status": "FAIL",
                "error": str(e)
            })
            self.verification_passed = False
            return False
    
    async def run_full_verification(self) -> bool:
        """Run the complete production verification suite."""
        logger.info(f"🚀 Starting production verification for {self.environment} environment (slot: {self.slot})")
        
        start_time = time.time()
        
        # Run all verification tests
        tests = [
            self.test_end_to_end_workflow(),
            self.test_load_performance(),
            self.test_database_integrity(),
            self.test_monitoring_and_alerting(),
            self.test_security_posture()
        ]
        
        # Execute all tests
        await asyncio.gather(*tests)
        
        execution_time = time.time() - start_time
        
        # Generate summary
        passed_count = len([r for r in self.results if r.get("status") == "PASS"])
        total_count = len(self.results)
        
        logger.info(f"\n📊 Production verification summary: {passed_count}/{total_count} tests passed")
        logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
        
        if self.verification_passed:
            logger.info("🎉 Production verification completed successfully! Deployment is ready for users.")
        else:
            logger.error("💥 Production verification failed! Please review and fix issues.")
        
        return self.verification_passed
    
    def save_results(self, output_file: str):
        """Save verification results to a file."""
        with open(output_file, 'w') as f:
            json.dump({
                "environment": self.environment,
                "slot": self.slot,
                "namespace": self.namespace,
                "timestamp": time.time(),
                "verification_passed": self.verification_passed,
                "results": self.results
            }, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Run production verification for AIVillage deployment")
    parser.add_argument("--environment", required=True, choices=["staging", "production"], help="Environment to verify")
    parser.add_argument("--slot", required=True, choices=["blue", "green"], help="Deployment slot")
    parser.add_argument("--output", default="production_verification_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    verifier = ProductionVerificationSuite(args.environment, args.slot)
    success = await verifier.run_full_verification()
    
    verifier.save_results(args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())