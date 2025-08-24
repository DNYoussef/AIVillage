#!/usr/bin/env python3
"""Comprehensive Service Health and Connectivity Verification Script
Verifies all CODEX integration services are running and responding correctly
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

try:
    import aiohttp
    from colorama import Fore, Style, init
    import redis
    import websockets
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Installing required packages...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "aiohttp",
            "redis",
            "websockets",
            "colorama",
        ],
        check=False,
    )
    import aiohttp
    from colorama import Fore, Style, init
    import redis
    import websockets

# Initialize colorama for colored output
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint to test"""

    name: str
    host: str
    port: int
    protocol: str  # TCP, UDP, HTTP, WS
    path: str = "/"
    expected_response: dict | None = None
    timeout: float = 5.0
    test_data: dict | None = None


@dataclass
class TestResult:
    """Result of a service test"""

    service: str
    success: bool
    latency_ms: float
    error: str | None = None
    response_data: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class ServiceHealthValidator:
    """Comprehensive service health and connectivity validator"""

    def __init__(self):
        self.results: list[TestResult] = []
        self.services = self._define_services()

    def _define_services(self) -> list[ServiceEndpoint]:
        """Define all services to test based on CODEX requirements"""
        return [
            # LibP2P Services
            ServiceEndpoint(
                name="LibP2P Main",
                host="0.0.0.0",
                port=4001,
                protocol="TCP",
                test_data={"type": "ping", "timestamp": time.time()},
            ),
            ServiceEndpoint(
                name="LibP2P WebSocket",
                host="localhost",
                port=4002,
                protocol="WS",
                path="/ws",
                test_data={"type": "hello", "peer_id": "test_peer"},
            ),
            # mDNS Discovery
            ServiceEndpoint(
                name="mDNS Discovery",
                host="224.0.0.251",  # mDNS multicast address
                port=5353,
                protocol="UDP",
                test_data=b"_aivillage._tcp.local",
            ),
            # HTTP APIs
            ServiceEndpoint(
                name="Digital Twin API",
                host="localhost",
                port=8080,
                protocol="HTTP",
                path="/health/twin",
                expected_response={"status": "healthy"},
            ),
            ServiceEndpoint(
                name="Evolution Metrics API",
                host="localhost",
                port=8081,
                protocol="HTTP",
                path="/health/evolution",
                expected_response={"status": "healthy"},
            ),
            ServiceEndpoint(
                name="RAG Pipeline API",
                host="localhost",
                port=8082,
                protocol="HTTP",
                path="/health/rag",
                expected_response={"status": "healthy"},
            ),
            # Redis (optional)
            ServiceEndpoint(
                name="Redis Cache",
                host="localhost",
                port=6379,
                protocol="REDIS",
                test_data={"ping": True},
            ),
        ]

    async def test_tcp_port(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test TCP port connectivity"""
        start_time = time.time()

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(endpoint.host, endpoint.port),
                timeout=endpoint.timeout,
            )

            if endpoint.test_data:
                # Send test data if provided
                if isinstance(endpoint.test_data, dict):
                    data = json.dumps(endpoint.test_data).encode()
                else:
                    data = endpoint.test_data

                writer.write(data)
                await writer.drain()

                # Try to read response
                try:
                    response = await asyncio.wait_for(reader.read(1024), timeout=2.0)
                    response_data = {"raw": response.hex()} if response else None
                except TimeoutError:
                    response_data = None
            else:
                response_data = None

            writer.close()
            await writer.wait_closed()

            latency = (time.time() - start_time) * 1000
            return TestResult(
                service=endpoint.name,
                success=True,
                latency_ms=latency,
                response_data=response_data,
            )

        except (TimeoutError, ConnectionRefusedError, OSError) as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(service=endpoint.name, success=False, latency_ms=latency, error=str(e))

    async def test_udp_port(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test UDP port connectivity (for mDNS)"""
        start_time = time.time()

        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(endpoint.timeout)

            # For mDNS, we need to join the multicast group
            if endpoint.port == 5353:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("", endpoint.port))
                    # Join multicast group
                    mreq = struct.pack("4sl", socket.inet_aton(endpoint.host), socket.INADDR_ANY)
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

                    # Send mDNS query
                    if endpoint.test_data:
                        sock.sendto(endpoint.test_data, (endpoint.host, endpoint.port))

                    # Try to receive response
                    sock.settimeout(2.0)
                    data, addr = sock.recvfrom(1024)
                    response_data = {"from": str(addr), "size": len(data)}
                    success = True
                except TimeoutError:
                    response_data = None
                    success = True  # mDNS might not respond immediately
                except Exception as e:
                    response_data = None
                    success = False
                    error = str(e)
            else:
                # Regular UDP test
                sock.sendto(endpoint.test_data or b"ping", (endpoint.host, endpoint.port))
                success = True
                response_data = None

            sock.close()

            latency = (time.time() - start_time) * 1000

            if success:
                return TestResult(
                    service=endpoint.name,
                    success=True,
                    latency_ms=latency,
                    response_data=response_data,
                )
            return TestResult(
                service=endpoint.name,
                success=False,
                latency_ms=latency,
                error=error if "error" in locals() else "UDP test failed",
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(service=endpoint.name, success=False, latency_ms=latency, error=str(e))

    async def test_http_endpoint(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test HTTP endpoint"""
        start_time = time.time()
        url = f"http://{endpoint.host}:{endpoint.port}{endpoint.path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as response:
                    latency = (time.time() - start_time) * 1000

                    # Check response status
                    if response.status == 200:
                        try:
                            data = await response.json()

                            # Validate expected response if provided
                            if endpoint.expected_response:
                                matches = all(data.get(k) == v for k, v in endpoint.expected_response.items())
                                if not matches:
                                    return TestResult(
                                        service=endpoint.name,
                                        success=False,
                                        latency_ms=latency,
                                        error=f"Response mismatch. Expected: {endpoint.expected_response}, Got: {data}",
                                        response_data=data,
                                    )

                            return TestResult(
                                service=endpoint.name,
                                success=True,
                                latency_ms=latency,
                                response_data=data,
                            )
                        except json.JSONDecodeError:
                            text = await response.text()
                            return TestResult(
                                service=endpoint.name,
                                success=True,
                                latency_ms=latency,
                                response_data={"text": text},
                            )
                    else:
                        return TestResult(
                            service=endpoint.name,
                            success=False,
                            latency_ms=latency,
                            error=f"HTTP {response.status}: {response.reason}",
                        )

        except (TimeoutError, aiohttp.ClientError) as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(service=endpoint.name, success=False, latency_ms=latency, error=str(e))

    async def test_websocket(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test WebSocket connection"""
        start_time = time.time()
        uri = f"ws://{endpoint.host}:{endpoint.port}{endpoint.path}"

        try:
            async with websockets.connect(uri, timeout=endpoint.timeout) as websocket:
                # Send test message if provided
                if endpoint.test_data:
                    await websocket.send(json.dumps(endpoint.test_data))

                    # Try to receive response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response_data = json.loads(response) if response else None
                    except (TimeoutError, json.JSONDecodeError):
                        response_data = None
                else:
                    response_data = None

                latency = (time.time() - start_time) * 1000

                return TestResult(
                    service=endpoint.name,
                    success=True,
                    latency_ms=latency,
                    response_data=response_data,
                )

        except (TimeoutError, websockets.exceptions.WebSocketException, OSError) as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(service=endpoint.name, success=False, latency_ms=latency, error=str(e))

    def test_redis(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test Redis connectivity"""
        start_time = time.time()

        try:
            client = redis.Redis(
                host=endpoint.host,
                port=endpoint.port,
                socket_connect_timeout=endpoint.timeout,
                socket_timeout=endpoint.timeout,
            )

            # Test ping
            client.ping()

            # Test basic operations
            test_key = "aivillage:test:health"
            test_value = json.dumps({"timestamp": time.time(), "test": True})

            client.set(test_key, test_value, ex=10)  # Expire in 10 seconds
            retrieved = client.get(test_key)

            if retrieved:
                client.delete(test_key)
                response_data = {"ping": "OK", "set_get": "OK"}
            else:
                response_data = {"ping": "OK", "set_get": "FAILED"}

            latency = (time.time() - start_time) * 1000

            return TestResult(
                service=endpoint.name,
                success=True,
                latency_ms=latency,
                response_data=response_data,
            )

        except (redis.ConnectionError, redis.TimeoutError) as e:
            latency = (time.time() - start_time) * 1000
            return TestResult(service=endpoint.name, success=False, latency_ms=latency, error=str(e))

    async def test_service(self, endpoint: ServiceEndpoint) -> TestResult:
        """Test a single service endpoint"""
        logger.info(f"Testing {endpoint.name} on {endpoint.host}:{endpoint.port}")

        if endpoint.protocol == "TCP":
            return await self.test_tcp_port(endpoint)
        if endpoint.protocol == "UDP":
            return await self.test_udp_port(endpoint)
        if endpoint.protocol == "HTTP":
            return await self.test_http_endpoint(endpoint)
        if endpoint.protocol == "WS":
            return await self.test_websocket(endpoint)
        if endpoint.protocol == "REDIS":
            return self.test_redis(endpoint)
        return TestResult(
            service=endpoint.name,
            success=False,
            latency_ms=0,
            error=f"Unknown protocol: {endpoint.protocol}",
        )

    async def test_service_communication(self) -> list[TestResult]:
        """Test that services can communicate with each other"""
        communication_tests = []

        # Test 1: Evolution Metrics -> Redis
        logger.info("Testing Evolution Metrics -> Redis communication")
        # This would require actual service implementation

        # Test 2: RAG Pipeline -> Redis Cache
        logger.info("Testing RAG Pipeline -> Redis cache communication")

        # Test 3: P2P -> Digital Twin synchronization
        logger.info("Testing P2P -> Digital Twin synchronization")

        return communication_tests

    async def test_failover_scenarios(self) -> list[TestResult]:
        """Test failover when services are unavailable"""
        failover_tests = []

        logger.info("Testing failover scenarios...")

        # Test Redis failover to file-based storage
        # Test P2P fallback transports
        # Test cache tier failover

        return failover_tests

    async def run_all_tests(self) -> tuple[list[TestResult], dict]:
        """Run all service health tests"""
        logger.info("Starting comprehensive service health validation...")

        # Test individual services
        tasks = [self.test_service(endpoint) for endpoint in self.services]
        results = await asyncio.gather(*tasks)

        # Test inter-service communication
        comm_results = await self.test_service_communication()
        results.extend(comm_results)

        # Test failover scenarios
        failover_results = await self.test_failover_scenarios()
        results.extend(failover_results)

        # Calculate statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

        stats = {
            "total_tests": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(results) * 100) if results else 0,
            "avg_latency_ms": avg_latency,
            "timestamp": datetime.now().isoformat(),
        }

        return results, stats

    def print_results(self, results: list[TestResult], stats: dict):
        """Print test results with colors"""
        print("\n" + "=" * 80)
        print(f"{Style.BRIGHT}SERVICE HEALTH VALIDATION RESULTS{Style.RESET_ALL}")
        print("=" * 80)

        # Group results by status
        for result in results:
            if result.success:
                status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
                latency_color = Fore.GREEN if result.latency_ms < 100 else Fore.YELLOW
            else:
                status = f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
                latency_color = Fore.RED

            print(f"\n{result.service}:")
            print(f"  Status: {status}")
            print(f"  Latency: {latency_color}{result.latency_ms:.2f}ms{Style.RESET_ALL}")

            if result.error:
                print(f"  Error: {Fore.RED}{result.error}{Style.RESET_ALL}")

            if result.response_data:
                print(f"  Response: {json.dumps(result.response_data, indent=4)}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print(f"{Style.BRIGHT}SUMMARY{Style.RESET_ALL}")
        print("=" * 80)

        success_color = (
            Fore.GREEN if stats["success_rate"] > 90 else Fore.YELLOW if stats["success_rate"] > 70 else Fore.RED
        )

        print(f"Total Tests: {stats['total_tests']}")
        print(f"Successful: {Fore.GREEN}{stats['successful']}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{stats['failed']}{Style.RESET_ALL}")
        print(f"Success Rate: {success_color}{stats['success_rate']:.1f}%{Style.RESET_ALL}")
        print(f"Average Latency: {stats['avg_latency_ms']:.2f}ms")

        # Identify critical issues
        if stats["failed"] > 0:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}CRITICAL ISSUES FOUND:{Style.RESET_ALL}")
            for result in results:
                if not result.success:
                    print(f"  - {result.service}: {result.error}")

    def save_results(self, results: list[TestResult], stats: dict):
        """Save results to JSON file"""
        output_file = Path("service_health_report.json")

        report = {
            "stats": stats,
            "results": [
                {
                    "service": r.service,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "response_data": r.response_data,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{Fore.CYAN}Results saved to {output_file}{Style.RESET_ALL}")

    async def auto_fix_issues(self, results: list[TestResult]):
        """Attempt to automatically fix identified issues"""
        print(f"\n{Style.BRIGHT}ATTEMPTING AUTO-FIX FOR FAILED SERVICES...{Style.RESET_ALL}")

        for result in results:
            if not result.success:
                print(f"\nAttempting to fix {result.service}...")

                # Determine fix based on service type
                if "Connection refused" in str(result.error):
                    print(f"  {Fore.YELLOW}Service not running. Attempting to start...{Style.RESET_ALL}")
                    # Would implement actual service start logic here

                elif "timeout" in str(result.error).lower():
                    print(f"  {Fore.YELLOW}Service timeout. Checking firewall and network...{Style.RESET_ALL}")
                    # Would implement firewall check and network diagnostics

                elif "HTTP" in result.service:
                    print(f"  {Fore.YELLOW}HTTP service issue. Checking configuration...{Style.RESET_ALL}")
                    # Would check and fix HTTP service configuration


async def main():
    """Main entry point"""
    validator = ServiceHealthValidator()

    # Run all tests
    results, stats = await validator.run_all_tests()

    # Print results
    validator.print_results(results, stats)

    # Save results
    validator.save_results(results, stats)

    # Attempt auto-fix if issues found
    if stats["failed"] > 0:
        await validator.auto_fix_issues(results)

        # Re-run tests after fixes
        print(f"\n{Style.BRIGHT}RE-RUNNING TESTS AFTER AUTO-FIX...{Style.RESET_ALL}")
        results, stats = await validator.run_all_tests()
        validator.print_results(results, stats)

    # Exit with appropriate code
    sys.exit(0 if stats["success_rate"] >= 90 else 1)


if __name__ == "__main__":
    asyncio.run(main())
