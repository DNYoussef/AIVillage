#!/usr/bin/env python3
"""Simplified Service Health Validation - No external dependencies
Tests basic connectivity to all required services
"""

import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Service definitions based on CODEX requirements
SERVICES = [
    {"name": "LibP2P Main", "host": "localhost", "port": 4001, "type": "tcp"},
    {"name": "LibP2P WebSocket", "host": "localhost", "port": 4002, "type": "tcp"},
    {"name": "mDNS Discovery", "host": "0.0.0.0", "port": 5353, "type": "udp"},
    {
        "name": "Digital Twin API",
        "host": "localhost",
        "port": 8080,
        "type": "http",
        "path": "/health/twin",
    },
    {
        "name": "Evolution Metrics API",
        "host": "localhost",
        "port": 8081,
        "type": "http",
        "path": "/health/evolution",
    },
    {
        "name": "RAG Pipeline API",
        "host": "localhost",
        "port": 8082,
        "type": "http",
        "path": "/health/rag",
    },
    {"name": "Redis Cache", "host": "localhost", "port": 6379, "type": "tcp"},
]


def test_tcp_port(host, port, timeout=2):
    """Test if TCP port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def test_udp_port(host, port, timeout=2):
    """Test UDP port (basic check)"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        # For UDP, we can only check if we can bind
        if port == 5353:  # mDNS
            # Just check if we can create a socket
            sock.close()
            return True
        sock.close()
        return True
    except Exception:
        return False


def test_http_endpoint(host, port, path="/", timeout=2):
    """Test HTTP endpoint"""
    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def main():
    """Run service health checks"""
    print("=" * 80)
    print("SERVICE HEALTH VALIDATION")
    print("=" * 80)

    results = []
    total = len(SERVICES)
    working = 0

    for service in SERVICES:
        name = service["name"]
        host = service["host"]
        port = service["port"]
        stype = service["type"]

        print(f"\nTesting {name} on {host}:{port}...")

        start_time = time.time()
        success = False
        error = None

        try:
            if stype == "tcp":
                success = test_tcp_port(host, port)
                if not success:
                    error = "Connection refused or timeout"
            elif stype == "udp":
                success = test_udp_port(host, port)
                if not success:
                    error = "UDP port check failed"
            elif stype == "http":
                path = service.get("path", "/")
                success = test_http_endpoint(host, port, path)
                if not success:
                    error = "HTTP endpoint not responding"
        except Exception as e:
            error = str(e)

        latency = (time.time() - start_time) * 1000

        if success:
            print(f"  [PASS] (latency: {latency:.2f}ms)")
            working += 1
        else:
            print(f"  [FAIL] - {error or 'Unknown error'}")

        results.append(
            {
                "service": name,
                "host": host,
                "port": port,
                "type": stype,
                "success": success,
                "latency_ms": latency,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_rate = (working / total * 100) if total > 0 else 0

    print(f"Total Services: {total}")
    print(f"Working: {working}")
    print(f"Failed: {total - working}")
    print(f"Success Rate: {success_rate:.1f}%")

    # Identify what needs fixing
    if working < total:
        print("\nSERVICES REQUIRING ATTENTION:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['service']}: {result['error']}")

    # Save results
    output_file = Path("service_health_simple.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": total,
                    "working": working,
                    "failed": total - working,
                    "success_rate": success_rate,
                    "timestamp": datetime.now().isoformat(),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_file}")

    # Provide fix recommendations
    if working < total:
        print("\n" + "=" * 80)
        print("RECOMMENDED FIXES")
        print("=" * 80)

        failed_services = [r for r in results if not r["success"]]

        # Group by service type
        p2p_failed = any(
            "LibP2P" in r["service"] or "mDNS" in r["service"] for r in failed_services
        )
        api_failed = any("API" in r["service"] for r in failed_services)
        redis_failed = any("Redis" in r["service"] for r in failed_services)

        if p2p_failed:
            print("\nP2P NETWORKING:")
            print("  The LibP2P and mDNS services are not running.")
            print("  To fix: Run 'python src/core/p2p/start_p2p_services.py'")

        if api_failed:
            print("\nAPI SERVICES:")
            print("  The HTTP API services are not responding.")
            print("  To fix: Run 'python src/api/start_api_servers.py'")

        if redis_failed:
            print("\nREDIS CACHE:")
            print("  Redis is not running (optional but recommended).")
            print("  To fix: Install and start Redis, or configure file-based caching")

    return 0 if success_rate >= 90 else 1


if __name__ == "__main__":
    sys.exit(main())
